/*
 * single-process-multi-tasks.c
 * 
 * Copyright 2021 chehw <hongwei.che@gmail.com>
 * 
 * The MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of 
 * this software and associated documentation files (the "Software"), to deal in 
 * the Software without restriction, including without limitation the rights to 
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to 
 * do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>
#include "opencl-context.h"
#include "opencl-kernel.h"
#include "utils.h"

#include <json-c/json.h>
#include <pthread.h>
#include <semaphore.h>

#ifndef FALSE
#define FALSE 	(0)
#define TRUE	(!(FALSE))
#endif

#define check_error(ret) do { 			\
		if(CL_SUCCESS == ret) break; 	\
		fprintf(stderr, "[ERROR]: %s(%d)::%s(): (err_code=%d), %s\n", \
			__FILE__, __LINE__, __FUNCTION__, 	\
			ret, opencl_error_to_string(ret)); 	\
		assert(CL_SUCCESS == ret);	\
	}while(0)


struct task_context;
typedef struct global_params
{
	void * user_data;
	void * priv;
	int verbose;
	
	int num_args;
	char ** non_option_args;
	
	const char * conf_file;
	json_object * jconfig;
	
	struct opencl_context * cl;
	struct opencl_platform * platform;
	const char * platform_name;
	
	struct opencl_device * device;
	cl_context ctx;
	cl_program program;
	
	int is_multi_processes;
	int num_tasks;
	struct task_context ** tasks;
	sem_t * tasks_sems;
	
	pthread_rwlock_t rw_mutex;
	int * tasks_status;
}global_params_t;

global_params_t * global_params_init(global_params_t * params, int argc, char ** argv, void ** user_data);
void global_params_cleanup(global_params_t * params);

static struct opencl_context g_cl_context[1];
int run_tasks(global_params_t * params);

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>


volatile int g_quit;
void on_signal(int sig)
{
	if(sig == SIGINT) {
		g_quit = 1;
		
		abort();
	}
	return;
}

int main(int argc, char **argv)
{
	signal(SIGINT, on_signal);
	// signal(SIGUSR1, on_signal);
	
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
	
	global_params_t * params = global_params_init(NULL, argc, argv, NULL);
	assert(params);
	
	opencl_context_t * cl = opencl_context_init(g_cl_context, params);
	assert(cl);
	assert(cl->num_platforms > 0);
	
	struct opencl_platform * platform = &cl->platforms[0];
	if(params->platform_name) platform = cl->get_platform_by_name_prefix(cl, params->platform_name);
	
	params->cl = cl;
	params->platform = platform;
	
	run_tasks(params);
	
	while(!g_quit) {
		sleep(1);
	}

	opencl_context_cleanup(cl);
	global_params_cleanup(params);
	return 0;
}

/*********************************************
 * run tasks
*********************************************/

typedef struct dim_3d
{
	size_t x;
	size_t y;
	size_t z;
}opencl_dim_3d;


struct task_context
{
	pthread_t thread_id;
	void * user_data;
	int index;
	
	global_params_t * params;
	json_object * jtask; // task config
	int quit;
	struct {
		pthread_mutex_t mutex;
		pthread_cond_t cond;
	}mc;
	
	cl_context ctx;
	struct opencl_device * device;
	struct opencl_function function[1];
	
	
	cl_command_queue queue;			// create a queue for the task to execute independent commands without requiring synchronization.
	size_t num_waiting_events;
	const cl_event * waiting_events;
	cl_event event;
	cl_int err_code;
	
	struct dim_3d offsets;
	struct dim_3d grid;		// global sizes
	struct dim_3d block;	// local sizes
	
	
	// callbacks 
	int (* on_init)(struct task_context * task, int task_index, void * user_data);
	int (* on_read_data)(struct task_context * task, int function_index, const char * kernel_name, void * user_data);

	// methods
	int (* run)(struct task_context * task);
};

static int task_run(struct task_context * task)
{
	return 0;
}

struct task_context * task_context_new(global_params_t * params, cl_context ctx, struct opencl_device * device)
{
	assert(params && ctx && device);
	
	struct task_context * task = calloc(1, sizeof(*task));
	assert(task);
	task->params = params;
	task->ctx = ctx;
	task->device = device;
	
	
	task->run = task_run;
	
	int rc = 0;
	rc = pthread_cond_init(&task->mc.cond, NULL);
	assert(0 == rc);
	rc = pthread_mutex_init(&task->mc.mutex, NULL);
	assert(0 == rc);
	
	return task;
}
void task_context_free(struct task_context * task)
{
	if(NULL == task) return;
	int rc = 0;
	if(task->event) {
		clSetUserEventStatus(task->event, CL_COMPLETE);
	}
	
	rc = pthread_mutex_lock(&task->mc.mutex);
	assert(0 == rc);
	task->quit = 1;
	pthread_cond_broadcast(&task->mc.cond);
	pthread_mutex_unlock(&task->mc.mutex);
	
	
	if(task->event) {
		clReleaseEvent(task->event);
		task->event = NULL;
	}
	
	if(task->queue) {
		clReleaseCommandQueue(task->queue);
		task->queue = NULL;
	}
	
	if(task->thread_id) {
		void * exit_code = NULL;
		int rc = pthread_join(task->thread_id, &exit_code);
		fprintf(stderr, "%s(tid=%p): thread exited with code %ld, rc = %d\n", 
			__FUNCTION__, (void *)(intptr_t)task->thread_id,
			(long)(intptr_t)exit_code, rc);
		task->thread_id = (pthread_t)0;
	}

	pthread_cond_destroy(&task->mc.cond);
	pthread_mutex_destroy(&task->mc.mutex);
	free(task);
}

// test data
#define NUM_TASKS (4)
#define ARRAY_SIZE (1024)
#define LOCAL_SIZE (256)
static const size_t s_array_lengths[NUM_TASKS] = {
	ARRAY_SIZE, 
	ARRAY_SIZE,
	ARRAY_SIZE * 2,
	ARRAY_SIZE * 2
};

static json_object * generate_dummy_config(void)
{
	/* 
	 * {
	 *   "tasks": [
	 *     { "dims": 3,
	 *       "offsets": [0, 0, 0],
	 *       "n": <array_length>,
	 *       "dependencies": [ tasks_index_list ],
	 *       "functions": [
	 *           "vec_add_scalar", // kernel_name_0
	 *           // ...
	 *           // kernel_name_n
	 * 
	 *       ]
	 *     },
	 *     { ... },
	 *   ]
	 * }
	*/
	json_object * jconfig = json_object_new_object();
	json_object * jtasks = json_object_new_array();
	json_object_object_add(jconfig, "tasks", jtasks);

	for(int i = 0; i < NUM_TASKS; ++i) {
		json_object * jtask = json_object_new_object();
		json_object_object_add(jtask, "dims", json_object_new_int(3));	// always use dim_3d
		json_object_object_add(jtask, "offsets", NULL);
		
		ssize_t array_length = s_array_lengths[i];
		assert(array_length > 0);
		json_object_object_add(jtask, "n", json_object_new_int64(array_length));
		
		json_object * jdependencies = json_object_new_array();
		json_object * jfunctions = json_object_new_array();
		
		switch(i) {
		case 0: case 1: 
			// no dependencies
			
			// functions
			json_object_array_add(jfunctions, json_object_new_string("vec_add_scalar"));
			break; 
		case 2: 
			// dependencies
			json_object_array_add(jdependencies, json_object_new_int(0));	// task 0
			json_object_array_add(jdependencies, json_object_new_int(1));	// task 1
			
			// functions
			json_object_array_add(jfunctions, json_object_new_string("vec_mul_scalar"));
			break;
		case 3:
			// dependencies
			json_object_array_add(jdependencies, json_object_new_int(2));	// task 2
			
			// functions
			json_object_array_add(jfunctions, json_object_new_string("vec_sum"));
			break;
		}
		json_object_object_add(jtask, "dependencies", jdependencies);
		json_object_object_add(jtask, "functions", jfunctions);
		json_object_array_add(jtasks, jtask);
	}
	return jconfig;
}

static void * process(void * user_data) 
{
	int rc = 0;
	cl_int ret = 0;
	struct task_context * task = user_data;
	assert(task && task->params);
	
	// run_task
	//~ pthread_mutex_lock(&task->mc.mutex); // lock mutex before cond_wait
	
	global_params_t * params = task->params;
	json_object * jtask = task->jtask;
	assert(jtask);
	
	assert(task->index >= 0 && task->index < params->num_tasks);
	sem_t * sems = params->tasks_sems;
	int * tasks_status = params->tasks_status;
	assert(sems && tasks_status);
	
	cl_context ctx = task->ctx;
	struct opencl_device * device = task->device;
	cl_program program = params->program;
	assert(ctx && device && program);
	
	json_object * jfunctions = NULL;
	json_bool ok = FALSE;
	ok = json_object_object_get_ex(jtask, "functions", &jfunctions);
	assert(ok && jfunctions);
	
#define MAX_FUNCTIONS (16)
	int num_functions = json_object_array_length(jfunctions);
	assert(num_functions > 0 && num_functions <= MAX_FUNCTIONS);
	
	// load kernels
	struct opencl_function *functions[MAX_FUNCTIONS];
	memset(functions, 0, sizeof(*functions));
	for(int i = 0; i < num_functions; ++i) {
		json_object * jkernel = json_object_array_get_idx(jfunctions, i);
		assert(jkernel);
		const char * kernel_name = json_object_get_string(jkernel);
		assert(kernel_name);
		
		functions[i] = opencl_function_init(NULL, program, kernel_name);
		assert(functions[i]);
		
		functions[i]->set_dims(functions[i], 3, (size_t *)&task->offsets, (size_t *)&task->grid, (size_t *)&task->block);
	}
	
	// init command queue
	cl_command_queue queue = task->queue;
	if(NULL == queue) {
		
		cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE 
				| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE	// kernels will be execute without order, use events to do sync
				| 0;	
				
		queue = clCreateCommandQueue(ctx, device->id, queue_props, &ret);
		assert(queue);
		task->queue = queue;
	}
	
	// init task
	if(task->on_init) task->on_init(task, task->index, task->user_data);
	
	sem_t * sem = &params->tasks_sems[task->index];
	assert(sem);
	while(!task->quit) {
		//~ rc = pthread_cond_wait(&task->mc.cond, &task->mc.mutex);
		//~ if(rc || task->quit) break;
		int cur_value = 0;
		rc = sem_getvalue(sem, &cur_value);
		printf("sems[%d] status: value=%d\n", task->index, cur_value);
		rc = sem_wait(sem);
		assert(0 == rc);
		if(task->quit) break;
		
		tasks_status[task->index] = 0;	// reset status
		for(int i = 0; i < num_functions; ++i) {
			struct opencl_function * function = functions[i];
			assert(function);
			
			// load data
			if(task->on_read_data) {
				task->on_read_data(task, i, function->kernel->name, task->user_data);
			}else {
				// todo: load default data for testing
				// opencl_function_set_args(function, num_args, size0, arg0, size1, arg1, ...);
			}
			// todo: execute
			//~ rc = function->execute(function, task->num_waiting_events, task->waiting_events, &task->event);
			//~ check_error(ret);
		}
		sleep(1);	// dummy process 
		
		// todo: add required synchronization for current task
		// ...
		tasks_status[task->index] = 1;
		
		// tasks sync 
		switch(task->index) {
		case 0: case 1:
			pthread_rwlock_wrlock(&params->rw_mutex);
			if(tasks_status[0] > 0 && tasks_status[1] > 0) {
				tasks_status[0] = -1;
				tasks_status[1] = -1;
				sem_post(&sems[2]);	// notify task2 to execute
			}
			pthread_rwlock_unlock(&params->rw_mutex);
			break;
		case 2:
			pthread_rwlock_wrlock(&params->rw_mutex);
			sem_post(&sems[3]);	// notify task2 to execute
			pthread_rwlock_unlock(&params->rw_mutex);
			break;
		case 3:
			pthread_rwlock_wrlock(&params->rw_mutex);
			sem_post(&sems[0]);	// notify task0 to execute
			sem_post(&sems[1]);	// notify task1 to execute
			pthread_rwlock_unlock(&params->rw_mutex);
			break;
		
		}
		
		if(task->quit) break;	// if quit signal has been set while processing
	}
	//~ pthread_mutex_unlock(&task->mc.mutex);
	pthread_exit((void *)(intptr_t)rc);
	
#if defined(_WIN32) || defined(WIN32)
	return (void *)(intptr_t)rc;
#endif
}

static int on_init_task(struct task_context * task, int task_index, void * user_data)
{
	fprintf(stderr, "[LOG]::%s(%p, %d, %p)\n", __FUNCTION__, task, task_index, user_data);
	
	return 0;
}

static int on_load_task_data(struct task_context * task, int function_index, const char * kernel_name, void * user_data)
{
	fprintf(stderr, "[LOG]::%s(%p, %d, %s, %p)\n", __FUNCTION__, task, function_index, kernel_name,  user_data);
	
	return 0;
}

int run_tasks(global_params_t * params)
{
	assert(params && params->cl && params->platform);
	int rc = 0;
	cl_int ret = 0;
	
	// todo: parse json config
	int device_index = 0;
	int device_type = CL_DEVICE_TYPE_GPU;
	
	opencl_context_t * cl = params->cl;
	struct opencl_platform * platform = params->platform;
	assert(cl && platform);

	rc = cl->load_devices(cl, device_type, platform);
	assert(0 == rc);
	
	struct opencl_device * device = &cl->devices[device_index];
	assert(device);
	cl_context_properties propertities[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform->id,
		0,
	};

	cl_context ctx = clCreateContext(propertities, 1, &device->id, NULL, NULL, &ret);
	check_error(ret);
	params->ctx = ctx;
	
	char * source = NULL;
	size_t cb_source = 0;
	const char * kernel_file = "kernels/kernels.cl";
	cb_source = load_file(kernel_file, &source);
	assert(cb_source != -1 && cb_source > 0);
	
	// build kernels
	cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&source, &cb_source, &ret);
	assert(program);
	free(source); source = NULL;
	
	ret = clBuildProgram(program, 1, &device->id, NULL, NULL, NULL);
	check_error(ret);
	params->program = program;

	json_object * jconfig = params->jconfig;
	if(NULL == jconfig) {
		jconfig = generate_dummy_config();
		assert(jconfig);
		params->jconfig = jconfig;
	}
	
	// load task settings
	json_object * jtasks = NULL;
	json_bool ok = FALSE;
	ok = json_object_object_get_ex(jconfig, "tasks", &jtasks);
	assert(ok && jtasks);
	
	int num_tasks = json_object_array_length(jtasks);
	assert(num_tasks > 0 && num_tasks <= NUM_TASKS);
	
	struct task_context ** tasks = calloc(num_tasks, sizeof(*tasks));
	sem_t * sems = calloc(num_tasks, sizeof(*sems));
	int * tasks_status = calloc(num_tasks, sizeof(*tasks_status));
	assert(tasks && sems && tasks_status);
	
	params->num_tasks = num_tasks;
	params->tasks = tasks;
	params->tasks_sems = sems;
	params->tasks_status = tasks_status;
	
	for(int i = 0; i < num_tasks; ++i) {
		// init semaphores
		switch(i) {
		case 0: case 1:
			rc = sem_init(&sems[i], 0, 1);
			break;
		case 2: case 3:
			rc = sem_init(&sems[i], 0, 0);
			break;
		default:
			break;
		}
		if(rc) perror("sem_init");
		assert(0 == rc);
		
		json_object * jtask = json_object_array_get_idx(jtasks, i);
		struct task_context * task = task_context_new(params, ctx, device);
		assert(task);
		tasks[i] = task;

		task->index = i;
		task->jtask = jtask;
		task->jtask = jtask;
		task->on_init = on_init_task;
		task->on_read_data = on_load_task_data;
		
		rc = pthread_create(&tasks[i]->thread_id, NULL, process, tasks[i]);
		assert(0 == rc);
	}

	return 0;
}

/*********************************************
 * global_params
*********************************************/
static void show_usuages(const char * exe_name)
{
	fprintf(stderr, "Usuage: %s \\\n"
		"--conf=<conf_file(default: conf/config.json)> \\\n"
		"--platform=<platform_name(default: nvidia)>\n", exe_name);
		
	return;
}
global_params_t * global_params_init(global_params_t * params, int argc, char ** argv, void ** user_data)
{
	if(NULL == params) params = calloc(1, sizeof(*params));
	else memset(params, 0, sizeof(*params));
	
	static struct option options[] = {
		{"conf", required_argument, 0, 'c'},
		{"platform", required_argument, 0, 'p'},
		{"verbose", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{NULL, }
	};
	
	int rc = 0;
	rc = pthread_rwlock_init(&params->rw_mutex, NULL);
	assert(0 == rc);
	
	const char * conf_file = NULL;
	const char * platform_name = NULL;
	int verbose = -1;
	while(1) {
		int option_index = 0;
		int c = getopt_long(argc, argv, "c:p:vh", options, &option_index);
		if(c == -1) break;
		switch(c) {
		case 'c': conf_file = optarg; break;
		case 'p': platform_name = optarg; break;
		case 'v': verbose = 1; break;
		case 'h': 
		default:
			show_usuages(argv[0]); 
			exit(c != 'h');
		}
	}
	
	if(optind < argc) {
		int num_args = argc - optind;
		assert(num_args > 0 && num_args < argc);
		char ** non_option_args = calloc(num_args, sizeof(*non_option_args));
		assert(non_option_args);
		for(int i = 0; i < num_args; ++i) {
			non_option_args[i] = argv[optind + i];
		}
		params->num_args = num_args;
		params->non_option_args = non_option_args;
	}
	
	if(conf_file) params->conf_file = conf_file;
	if(platform_name) params->platform_name = platform_name;
	if(verbose >= 0) params->verbose = verbose;
	
	if(params->conf_file) {
		params->jconfig = json_object_from_file(params->conf_file);
		assert(params->jconfig);
	}
	
	return params;
}

void global_params_cleanup(global_params_t * params)
{
	if(NULL == params) return;

	pthread_rwlock_wrlock(&params->rw_mutex);
	if(params->non_option_args) free(params->non_option_args);
	params->non_option_args = NULL;
	params->num_args = 0;

	sem_t * sems = params->tasks_sems;
	struct task_context ** tasks = params->tasks;
	if(sems || tasks) {
		assert(sems && tasks);
		for(int i = 0; i < params->num_tasks; ++i) {
			if(tasks[i]) tasks[i]->quit = 1;
			sem_post(&sems[i]); // cancel sem_wait(ing) jobs
			
			if(tasks[i]) {
				task_context_free(tasks[i]);
				tasks[i] = NULL;
			}
			if(!params->is_multi_processes) sem_destroy(&sems[i]);
		}
		free(sems);
		free(tasks);
	}
	free(params->tasks_status);
	params->num_tasks = 0;
	params->tasks = NULL;
	params->tasks_sems = NULL;
	params->tasks_status = NULL;

	if(params->jconfig) {
		json_object_put(params->jconfig);
		params->jconfig = NULL;
	}
	
	if(params->program) {
		clReleaseProgram(params->program);
		params->program = NULL;
	}
	
	pthread_rwlock_unlock(&params->rw_mutex);
	pthread_rwlock_destroy(&params->rw_mutex);
	return;
}
