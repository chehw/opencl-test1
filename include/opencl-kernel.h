#ifndef OPENCL_KERNEL_H_
#define OPENCL_KERNEL_H_

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h>

#include <CL/cl.h>
#include "opencl-context.h"

struct opencl_program
{
	cl_program prog;
	cl_context ctx;
	
	int build_status;
	
	size_t num_devices;
	cl_device_id * device_ids;

	int (* load_sources)(struct opencl_program * program, size_t num_sources, const char ** sources, const size_t * lengths);
	int (* load_binaries)(struct opencl_program * program, const size_t * lengths, const unsigned char ** binaries); 
	int (* load_builtin_kernels)(struct opencl_program * program, const char * kernel_names); // kernel_names: A semi-colon separated list of built-in kernel names.
	
	int (* compile)(struct opencl_program * program, const char * options, size_t num_headers, const cl_program * headers, const char ** header_names);
	int (* link)(struct opencl_program * program, const char * options, size_t num_input_programs, const cl_program * input_programs);
	
	// compile and link
	int (* build)(struct opencl_program * program, const char * options);
};
struct opencl_program * opencl_program_init(struct opencl_program * program, cl_context ctx, size_t num_devices, const cl_device_id * device_ids);
void opencl_program_cleanup(struct opencl_program * program);
ssize_t opencl_program_get_build_log(struct opencl_program * program, cl_device_id device_id, char *p_build_log, size_t build_log_size);

struct opencl_kernel
{
	cl_kernel _kernel;
	char name[100];
	size_t num_args;
	size_t * sizes;
	void ** args;
};
struct opencl_kernel * opencl_kernel_init(struct opencl_kernel * kernel, cl_program prog, const char * kernel_name);
void opencl_kernel_cleanup(struct opencl_kernel * kernel);
int opencl_kernel_set_args(struct opencl_kernel * kernel, size_t num_args, ... /* size_t size1, void * arg1, ...*/ );

struct opencl_function
{
	struct opencl_kernel kernel[1]; // base object
	#define opencl_function_set_args(func, num_args, ...) opencl_kernel_set_args((struct opencl_kernel *)func, num_args, __VA_ARGS__)

	size_t work_dim;
	size_t * global_offsets;
	size_t * global_sizes;	// ==> cuda::{grid.x, grid.y, grid.z} 
	size_t * local_sizes;	// ==> cuda::{block.x, block.y, block.z}
	
	cl_command_queue queue;
	cl_event event;
	
	int (* set_dims)(struct opencl_function * function, size_t work_dim, const size_t * global_offsets, const size_t * global_sizes, const size_t * local_sizes);
	int (* execute)(struct opencl_function * function, size_t num_waiting_events, const cl_event * waiting_events, cl_event * event);
};
struct opencl_function * opencl_function_init(struct opencl_function * function, const cl_program program, const char * kernel_name);
void opencl_function_cleanup(struct opencl_function * function);

struct opencl_event_list
{
	size_t size;
	size_t length;
	const cl_event * events;
	
	int (* add)(struct opencl_event_list * list, const cl_event * event);
	const cl_event *  (* remove)(struct opencl_event_list * list, int index);
};
struct opencl_event_list * opencl_event_list_init(struct opencl_event_list * list, size_t size);
void opencl_event_list_cleanup(struct opencl_event_list * list);

struct opencl_buffer
{
	cl_mem gpu_data;
	size_t size;
	cl_mem_flags flags;
	
	void * cpu_data;
	void (* on_free_cpu_data)(void *);
	
	const struct opencl_event_list * waiting_list;
	cl_event event;
	cl_int err_code;
};

struct opencl_buffer * opencl_buffer_init(struct opencl_buffer * buf, cl_context ctx, cl_mem_flags flags, size_t size, const void * cpu_data);
void opencl_buffer_cleanup(struct opencl_buffer * buf);
void * opencl_buffer_enqueue_read(struct opencl_buffer * buf, cl_command_queue queue, cl_bool blocking, size_t offset, size_t length, const void * cpu_data);
int opencl_buffer_enqueue_write(struct opencl_buffer * buf, cl_command_queue queue, cl_bool blocking, size_t offset, size_t length);

#ifdef __cplusplus
}
#endif
#endif
