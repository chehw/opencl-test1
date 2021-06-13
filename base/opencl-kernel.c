/*
 * opencl-kernel.c
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

#include <stdarg.h>
#include "opencl-kernel.h"


#define check_error(ret) do { 			\
		if(CL_SUCCESS == ret) break; 	\
		fprintf(stderr, "[ERROR]: %s(%d)::%s(): (err_code=%d), %s\n", \
			__FILE__, __LINE__, __FUNCTION__, 	\
			ret, opencl_error_to_string(ret)); 	\
		assert(CL_SUCCESS == ret);	\
	}while(0)

/* *
struct opencl_program
* */
static int program_load_sources(struct opencl_program * program, size_t num_sources, const char ** sources, const size_t * lengths)
{
	assert(program && program->ctx);
	assert(num_sources > 0 && sources && lengths);
	
	cl_int ret = 0;
	cl_context ctx = program->ctx;
	
	if(program->prog) {
		clReleaseProgram(program->prog);
		program->prog = NULL;
	}
	
	program->build_status = CL_BUILD_NONE;
	program->prog = clCreateProgramWithSource(ctx, num_sources, sources, lengths, &ret);
	
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
			
		program->build_status = CL_BUILD_ERROR;
		return -1;
	}
	return 0;
}

static int program_load_binaries(struct opencl_program * program, const size_t * lengths, const unsigned char ** binaries)
{
	assert(program && program->ctx);
	assert(program->num_devices > 0 && program->device_ids);
	assert(lengths > 0 && binaries);
	
	cl_int ret = 0;
	cl_context ctx = program->ctx;
	
	if(program->prog) {
		clReleaseProgram(program->prog);
		program->prog = NULL;
	}
	
	cl_int bin_status[program->num_devices];	// c99
	memset(bin_status, 0, sizeof(bin_status));
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	program->prog = clCreateProgramWithBinary(ctx, 
		program->num_devices, program->device_ids,
		lengths, binaries, 
		bin_status,
		&ret);
		
	for(int i = 0;i < program->num_devices; ++i) {
		if(bin_status[i] != CL_SUCCESS) {
			fprintf(stderr, "[WARNING]::%s(%d)::%s(): invalid program binary on device[%d]: %s\n", 
				__FILE__, __LINE__, __FUNCTION__, 
				i, opencl_error_to_string(ret));
		}
	}

	
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));

		program->build_status = CL_BUILD_ERROR;
		return -1;
	}
	
	program->build_status = CL_BUILD_SUCCESS;
	return 0;
}

static int program_load_builtin_kernels(struct opencl_program * program, const char * kernel_names)
{
	assert(program && program->ctx);
	assert(program->num_devices > 0 && program->device_ids);
	assert(kernel_names);
	
	cl_int ret = 0;
	cl_context ctx = program->ctx;
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	program->prog = clCreateProgramWithBuiltInKernels(ctx, 
		program->num_devices, program->device_ids,
		kernel_names,
		&ret);
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
		
		program->build_status = CL_BUILD_ERROR;
		return -1;
	}
	
	program->build_status = CL_BUILD_SUCCESS;
	return 0;
}

static int program_compile(struct opencl_program * program, const char * options, size_t num_headers, const cl_program * headers, const char ** header_names)
{
	assert(program && program->ctx);
	assert(program->num_devices > 0 && program->device_ids);
	cl_int ret = 0;
	cl_program prog = program->prog;
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	ret = clCompileProgram(prog, program->num_devices, program->device_ids,
		options,
		num_headers, headers, header_names,
		NULL, NULL);
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
			
		program->build_status = CL_BUILD_ERROR;
		return -1;
	}
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	return 0;
}

static int program_link(struct opencl_program * program, const char * options, size_t num_input_programs, const cl_program * input_programs)
{
	assert(program && program->ctx);
	assert(program->num_devices > 0 && program->device_ids);
	cl_int ret = 0;
	cl_context ctx = program->ctx;
	
	if(program->prog) {
		clReleaseProgram(program->prog);
		program->prog = NULL;
	}
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	program->prog = clLinkProgram(ctx, program->num_devices, program->device_ids, 
		options,
		num_input_programs, input_programs, 
		NULL, NULL, 
		&ret);
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
			
		program->build_status = CL_BUILD_ERROR;
		return -1;
	}
	
	program->build_status = CL_BUILD_SUCCESS;
	return 0;
}

static int program_build(struct opencl_program * program, const char * options)
{
	assert(program && program->ctx);
	assert(program->num_devices > 0 && program->device_ids);
	cl_int ret = 0;
	cl_program prog = program->prog;
	
	program->build_status = CL_BUILD_IN_PROGRESS;
	
	ret = clBuildProgram(prog, program->num_devices, program->device_ids, options, NULL, NULL);
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
			
		program->build_status = CL_BUILD_ERROR;
	}
	
	program->build_status = CL_BUILD_SUCCESS;
	return 0;
}

ssize_t opencl_program_get_build_log(struct opencl_program * program, cl_device_id device_id, char * build_log, size_t build_log_size)
{
	assert(program);
	cl_int ret = 0;
	cl_program prog = program->prog;
	
	if(NULL == prog) return -1;
	
	size_t cb_log = 0;
	ret = clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, &cb_log);
	if(ret != CL_SUCCESS) {
		fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
		return -1;
	}
	return cb_log;
}


struct opencl_program * opencl_program_init(struct opencl_program * program, cl_context ctx, size_t num_devices, const cl_device_id * device_ids)
{
	if(NULL == program) {
		program = calloc(1, sizeof(*program));
		assert(program);
	}else memset(program, 0, sizeof(*program));
	
	program->load_sources = program_load_sources;
	program->load_binaries = program_load_binaries;
	program->load_builtin_kernels = program_load_builtin_kernels;
	program->compile = program_compile;
	program->link = program_link;
	program->build = program_build;
	
	program->ctx = ctx;
	program->build_status = CL_BUILD_NONE;
	
	if(ctx) {
		cl_int ret = clRetainContext(ctx); // add_ref
		if(ret != CL_SUCCESS) {
			fprintf(stderr, "[ERROR]::%s(%d)::%s(): %s\n", 
			__FILE__, __LINE__, __FUNCTION__, 
			opencl_error_to_string(ret));
		}
		assert(CL_SUCCESS == ret);
	}
	
	if(num_devices > 0 && device_ids) {
		cl_device_id * ids = calloc(num_devices, sizeof(*ids));
		assert(ids);
		
		program->num_devices = num_devices;
		memcpy(ids, device_ids, sizeof(*ids) * num_devices);
		
		program->num_devices = num_devices;
		program->device_ids = ids;
	}
	
	return program;
}

void opencl_program_cleanup(struct opencl_program * program)
{
	if(NULL == program) return;
	cl_program prog = program->prog;
	program->prog = NULL;
	if(prog) clReleaseProgram(prog);
	
	cl_device_id * ids = program->device_ids;
	program->device_ids = NULL;
	if(ids) free(ids);

	program->num_devices = 0;
	program->build_status = CL_BUILD_NONE;
	
	cl_context ctx = program->ctx;
	program->ctx = NULL;
	if(ctx) clReleaseContext(ctx);	// unref
	
	return;
}

/* *
struct opencl_kernel
* */
struct opencl_kernel * opencl_kernel_init(struct opencl_kernel * kernel, cl_program prog, const char * kernel_name)
{
	assert(prog && kernel_name);
	
	if(NULL == kernel) {
		kernel = calloc(1, sizeof(*kernel));
		assert(kernel);
	}else
	{
		memset(kernel, 0, sizeof(*kernel));
	}
	
	cl_int ret = 0;
	kernel->_kernel = clCreateKernel(prog, kernel_name, &ret);
	check_error(ret);
	assert(ret == CL_SUCCESS);
	
	strncpy(kernel->name, kernel_name, sizeof(kernel->name));
	return kernel;
}

void opencl_kernel_cleanup(struct opencl_kernel * kernel)
{
	if(NULL == kernel) return;
	if(kernel->_kernel) {
		clReleaseKernel(kernel->_kernel);
		kernel->_kernel = NULL;
	}
	if(kernel->sizes) free(kernel->sizes);
	if(kernel->args) free(kernel->args);
	
	kernel->num_args = 0;
	kernel->sizes = NULL;
	kernel->args = NULL;
	return;
}
int opencl_kernel_set_args(struct opencl_kernel * kernel, size_t num_args, ... /* size_t size1, void * arg1, ...*/ )
{
	assert(kernel && num_args > 0);
	
	va_list ap;
	va_start(ap, num_args);
	
	kernel->num_args = num_args;
	kernel->sizes = realloc(kernel->sizes, sizeof(*kernel->sizes) * num_args);
	kernel->args = realloc(kernel->args, sizeof(*kernel->args) * num_args);
	assert(kernel->sizes && kernel->args);
	
	for(size_t i = 0; i < num_args; ++i) {
		kernel->sizes[i] = va_arg(ap, size_t);
		kernel->args[i] = va_arg(ap, void *);
	}
	va_end(ap);
	return 0;
}

/* *
struct opencl_function
* */
static int function_set_dims(struct opencl_function * function, size_t work_dim, const size_t * global_offsets, const size_t * global_sizes, const size_t * local_sizes)
{
	return 0;
}
static int function_execute(struct opencl_function * function, size_t num_waiting_events, const cl_event * waiting_events, cl_event * event)
{
	return 0;
}

struct opencl_function * opencl_function_init(struct opencl_function * function, const cl_program program, const char * kernel_name)
{
	if(NULL == function) {
		function = calloc(1, sizeof(*function));
		assert(function);
	}else {
		memset(function, 0, sizeof(*function));
	}
	
	function->set_dims = function_set_dims;
	function->execute = function_execute;
	
	if(program && kernel_name) {
		struct opencl_kernel * kernel = opencl_kernel_init((struct opencl_kernel *)function, program, kernel_name);
		assert(kernel && kernel == function->kernel);
	}
	
	return function;
}

void opencl_function_cleanup(struct opencl_function * function)
{
	opencl_kernel_cleanup(function->kernel);
	
	if(function->global_offsets) free(function->global_offsets);
	if(function->global_sizes) free(function->global_sizes);
	if(function->local_sizes) free(function->local_sizes);
	function->work_dim = 0;
	function->global_offsets = NULL;
	function->global_sizes = NULL;
	function->local_sizes = NULL;
	
	if(function->event) {
		clSetUserEventStatus(function->event, CL_COMPLETE);
		clReleaseEvent(function->event);
		function->event = NULL;
	}
	return;
}

/* *
struct opencl_buffer
{
	cl_mem gpu_data;
	size_t size;
	cl_mem_flags flags;
	
	void * cpu_data;
	void (* on_cpu_data_free)(void *);
	
	const opencl_event_list * waiting_list;
	cl_event event;
	cl_int err_code;
};
* */

struct opencl_buffer * opencl_buffer_init(struct opencl_buffer * buf, cl_context ctx, cl_mem_flags flags, size_t size, const void * cpu_data)
{
	if(NULL == buf) buf = calloc(1, sizeof(*buf));
	else memset(buf, 0, sizeof(*buf));
	assert(buf);
	
	assert(size > 0);
	buf->gpu_data = clCreateBuffer(ctx, flags, size, (void *)cpu_data, &buf->err_code);
	check_error(buf->err_code);
	
	return buf;
}
void opencl_buffer_cleanup(struct opencl_buffer * buf)
{
	if(NULL == buf) return;
	
	buf->waiting_list = NULL;
	if(buf->event) {
		clSetUserEventStatus(buf->event, CL_COMPLETE);
		buf->event = NULL;
	}
	
	if(buf->cpu_data && buf->on_free_cpu_data) {
		buf->on_free_cpu_data(buf->cpu_data);
	}
	buf->cpu_data = NULL;
	
	if(buf->gpu_data) {
		clReleaseMemObject(buf->gpu_data);
		buf->gpu_data = NULL;
	}
	
	buf->size = 0;
	buf->flags = 0;
	return;
}

void * opencl_buffer_enqueue_read(struct opencl_buffer * buf, cl_command_queue queue, cl_bool blocking, size_t offset, size_t length, const void * cpu_data)
{
	assert(buf);
	assert(length > 0);
	
	
	// todo:
	return NULL;
}
int opencl_buffer_enqueue_write(struct opencl_buffer * buf, cl_command_queue queue, cl_bool blocking, size_t offset, size_t length)
{
	// todo:
	return -1;
}
