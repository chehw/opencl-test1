/*
 * test1.c
 * 
 * Copyright 2021 htcch <htcch@DESKTOP-PSIPCOL>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <CL/cl.h>
#include <stdint.h>
#include <inttypes.h>

#include "opencl-context.h"

#define check_error(ret) do { 			\
		if(CL_SUCCESS == ret) break; 	\
		fprintf(stderr, "[ERROR]: %s(%d)::%s(): (err_code=%d), %s\n", \
			__FILE__, __LINE__, __FUNCTION__, 	\
			ret, opencl_error_to_string(ret)); 	\
		assert(CL_SUCCESS == ret);	\
	}while(0)
	
int run_test(int num_sub_devices, cl_device_id * sub_device_ids, opencl_context_t * cl);

int main(int argc, char **argv)
{
#if defined(WIN32) || defined(_WIN32)
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
#endif

	int aligned = (63 + 15) & ~15;
	assert(aligned == 64);
	
	int rc = 0;
	opencl_context_t * cl = opencl_context_init(NULL, NULL);
	assert(cl);
	
	struct opencl_platform * platform = cl->get_platform_by_name_prefix(cl, "NVIDIA");
	assert(platform);
	
	rc = cl->load_devices(cl, 0, platform);
	assert(0 == rc);
	
	assert(cl->num_devices > 0);
	struct opencl_device * device = &cl->devices[0];
	assert(device && device->id && device->is_available);
	
	#define NUM_SUB_DEVICES (4)
	cl_device_id sub_device_ids[NUM_SUB_DEVICES] = { NULL };
	cl_uint num_sub_devices = 0;
	cl_int ret;
	
	printf("max_compute_units: %d\n", (int)device->max_compute_units);
	printf("max_sub_devices: %d\n", (int)device->max_sub_devices);
	if(device->max_sub_devices > 1) // the device can be partitioned
	{
		cl_uint num_sub_devices = NUM_SUB_DEVICES;
		if(num_sub_devices > device->max_sub_devices) num_sub_devices = device->max_sub_devices;
		
		int num_compute_units_per_sub_device = device->max_compute_units / num_sub_devices;
		assert(num_compute_units_per_sub_device >= 1);
		
		cl_device_partition_property propertities[] = {
				CL_DEVICE_PARTITION_BY_COUNTS, // use CL_DEVICE_PARTITION_BY_COUNTS to partition the device with diffrent size
				1,
				CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
				0
		};
		
		ret = clCreateSubDevices(device->id, propertities, NUM_SUB_DEVICES, sub_device_ids, &num_sub_devices);
		check_error(ret);
		
		assert(num_sub_devices == NUM_SUB_DEVICES);
		run_test(num_sub_devices, sub_device_ids, cl);
		
	}else {
		cl_uint num_devices = cl->num_devices;
		assert(num_devices > 0);
		
		cl_device_id device_ids[num_devices]; // C99 vla
		memset(device_ids, 0, sizeof(device_ids));
		for(cl_uint i = 0; i < num_devices; ++i) {
			device_ids[i] = cl->devices[i].id;
		}
		run_test(num_devices, device_ids, cl);
	}

// cleanup:
	for(cl_uint i = 0; i < num_sub_devices; ++i) {
		if(sub_device_ids[i]) {
			clReleaseDevice(sub_device_ids[i]);
			sub_device_ids[i] = NULL;
		}
	}
	
	opencl_context_cleanup(cl);
	free(cl);
	
	return 0;
}

/*******************************
 * utils
*******************************/
#include <sys/stat.h>
#include <unistd.h>

ssize_t load_file(const char * filename, char ** p_data)
{
	int rc = 0;
	struct stat st[1];
	memset(st, 0, sizeof(st));
	rc = stat(filename, st);
	if(rc) return -1;
	if((st->st_mode & S_IFMT) != S_IFREG) return -1;
	if(st->st_size == 0) return 0;
	
	if(NULL == p_data) return (st->st_size + 1); // return buffer size
	
	FILE * fp = fopen(filename, "rb");
	assert(fp);
	
	char * data = *p_data;
	if(NULL == data) {
		data = malloc(st->st_size + 1);
		assert(data);
		*p_data = data;
	}
	ssize_t length = fread(data, 1, st->st_size, fp);
	assert(length == st->st_size);
	data[length] = '\0';
	fclose(fp);
	
	return length;
}

/*******************************
 * run tests
*******************************/

static void CL_CALLBACK on_notify_create_context_error(const char * err_info, const void * priv_data, size_t cb, void * user_data)
{
	fprintf(stderr, "%s(%s, %p, %lu)\n", __FUNCTION__, err_info, priv_data, (unsigned long)cb);
	return;
}

int run_test(int num_devices, cl_device_id * device_ids, opencl_context_t * cl)
{
	cl_context_properties propertities[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)cl->platforms[0].id,
		//~ CL_CONTEXT_INTEROP_USER_SYNC,
		//~ CL_TRUE,
		0,
	};
	cl_int ret = CL_SUCCESS;
	cl_context ctx = clCreateContext(propertities, num_devices, device_ids,
		on_notify_create_context_error, NULL, &ret);
	check_error(ret);
	
#define NUM_COMMAND_QUEUE (4)
/**************************************************************************************************
 *                                                |
 * Queue_0([IN] buf_0, [OUT] buffer[2]:offset_0)  |  Queue_1([IN] buf_1, [OUT]buffer[2]:offset_1)  
 *                                                |
 * -------------------------------------------------------------------------------------------------
 *                                                |
 *                                                V
 *                                Queue_2([IN] buf_2, [OUT] buf_3) 
 *                                                |
 *                                                V
 *                                  Queue_2([IN] buf_3, [OUT]output_value)
***************************************************************************************************/
	
	// step 1. create command queues
	cl_command_queue queues[NUM_COMMAND_QUEUE];
	memset(queues, 0, sizeof(queues));
	
	cl_device_id device = device_ids[0];	// test devices[0] only
	for(int i = 0; i < NUM_COMMAND_QUEUE; ++i) {
		ret = CL_SUCCESS;
		int queue_props = CL_QUEUE_PROFILING_ENABLE 
		//	| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE	// kernels will be execute without order, use events to do sync
			| 0;	
			
		queues[i] = clCreateCommandQueue(ctx, device, queue_props, &ret);
		check_error(ret);
	}
	
	// step 2. create buffers
	cl_mem buffers[NUM_COMMAND_QUEUE];	
	memset(buffers, 0, sizeof(buffers));
	
#define ARRAY_SIZE (1024)
	static size_t array_lengths[] = {
		[0] = ARRAY_SIZE,
		[1] = ARRAY_SIZE,
		[2] = ARRAY_SIZE * 2,
		[3] = ARRAY_SIZE * 2,
	};
	static cl_mem_flags flags[] = {
		[0] = CL_MEM_READ_ONLY,
		[1] = CL_MEM_READ_ONLY,
		[2] = CL_MEM_READ_WRITE,
		[3] = CL_MEM_READ_WRITE,
	};
	
	for(int i = 0; i < NUM_COMMAND_QUEUE; ++i) {
		ret = CL_SUCCESS;
		buffers[i] = clCreateBuffer(ctx, flags[i], array_lengths[i] * sizeof(float), NULL, &ret);
		check_error(ret);
		assert(buffers[i]);
	}
	
	// step 3. create program from source file
	const char * kernels_file = "kernels/kernels.cl";
	char * sources = NULL;
	size_t length = load_file(kernels_file, &sources);
	printf("length: %d\n", (int)length);
	
	assert(length != -1 && length > 0 && sources);
	
	cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&sources, &length, &ret);
	check_error(ret);
	assert(program);
	
	ret = clBuildProgram(program, num_devices, device_ids, NULL, NULL, NULL);
	check_error(ret);
	
	/* 
	 * step 4. load kernels and set args
	 * __kernel void vec_mul_scalar(__global float * Y, __global float * X, __const float a);
	 * __kernel void vec_add_scalar(__global float * Y, __global const float * X, __const float a, __const int y_offset);
	 * __kernel void vec_sum(__const int n, __global float * A, __local float * partials, __global float * result);
	*/
	cl_kernel vec_add_scalar_0 = clCreateKernel(program, "vec_add_scalar", &ret);	// for queue[0]
	check_error(ret);
	cl_kernel vec_add_scalar_1 = clCreateKernel(program, "vec_add_scalar", &ret);	// for queue[1]
	check_error(ret);
	
	cl_kernel vec_mul_scalar = clCreateKernel(program, "vec_mul_scalar", &ret);		// for queue[2]
	check_error(ret);
	
	cl_kernel vec_sum = clCreateKernel(program, "vec_sum", &ret);					// for queue[3]
	check_error(ret);
	
	cl_float a_0 = 1.0f;	// for queue_0
	cl_float a_1 = 2.0f;	// for queue_1
	cl_float a_2 = 3.0f;	// for queue_2

	cl_int offset_0 = 0;
	cl_int offset_1 = ARRAY_SIZE;	
	// queue_0
	clSetKernelArg(vec_add_scalar_0, 0, sizeof(cl_mem), &buffers[2]);	// Y
	clSetKernelArg(vec_add_scalar_0, 1, sizeof(cl_mem), &buffers[0]);	// X
	clSetKernelArg(vec_add_scalar_0, 2, sizeof(cl_float), &a_0);		// a
	clSetKernelArg(vec_add_scalar_0, 3, sizeof(cl_int), &offset_0);		// offset
	
	// queue_1
	clSetKernelArg(vec_add_scalar_1, 0, sizeof(cl_mem), &buffers[2]);	// Y
	clSetKernelArg(vec_add_scalar_1, 1, sizeof(cl_mem), &buffers[1]);	// X
	clSetKernelArg(vec_add_scalar_1, 2, sizeof(cl_float), &a_1);		// a
	clSetKernelArg(vec_add_scalar_1, 3, sizeof(cl_int), &offset_1);		// offset
	
	// queue_2
	clSetKernelArg(vec_mul_scalar, 0, sizeof(cl_mem), &buffers[3]);		// Y
	clSetKernelArg(vec_mul_scalar, 1, sizeof(cl_mem), &buffers[2]);		// X
	clSetKernelArg(vec_mul_scalar, 2, sizeof(cl_float), &a_2);			// a
	
	// queue_3
	cl_int n = (cl_int)array_lengths[3];
	size_t local_size = 256;
	assert(local_size < n && (n % local_size == 0));
	size_t num_groups = n / local_size;

	cl_float * results = calloc(num_groups, sizeof(float));	// for queue_3
	cl_mem mem_results = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, num_groups * sizeof(float), NULL, &ret);
	check_error(ret);
	
	clSetKernelArg(vec_sum, 0, sizeof(cl_int), &n);						// n
	clSetKernelArg(vec_sum, 1, sizeof(cl_mem), &buffers[3]);			// A
	clSetKernelArg(vec_sum, 2, local_size * sizeof(float), NULL);		// __local size for reduction 
	clSetKernelArg(vec_sum, 3, sizeof(cl_mem), &mem_results);			// result
	
	// step 5. init inputs and outputs buffer on the host
	float * buf_0 = calloc(array_lengths[0], sizeof(float));
	float * buf_1 = calloc(array_lengths[1], sizeof(float));
	float * buf_2 = calloc(array_lengths[2], sizeof(float));
	float * buf_3 = calloc(array_lengths[3], sizeof(float));
	assert(buf_0 && buf_1 && buf_2 && buf_3);
	
	for(int i = 0; i < ARRAY_SIZE; ++i) {
		buf_0[i] = i + 1;
		buf_1[i] = ARRAY_SIZE - i - 1;
	}
	
	// step 6. copy inputs buffer from host to GPU memory
	ret = clEnqueueWriteBuffer(queues[0], buffers[0], CL_TRUE, 
		0, array_lengths[0] * sizeof(float), buf_0, 
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueWriteBuffer(queues[1], buffers[1], CL_TRUE, 
		0, array_lengths[1] * sizeof(float), buf_1, 
		0, NULL, NULL);
	check_error(ret);
	
	// step 7. execute kernels
	ret = clEnqueueNDRangeKernel(queues[0], vec_add_scalar_0, 1,
		NULL, 
		(size_t[]){array_lengths[0], 1, 1},
		(size_t[]){local_size, 1, 1},
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueNDRangeKernel(queues[1], vec_add_scalar_1, 1,
		NULL, 
		(size_t[]){array_lengths[1], 1, 1},
		(size_t[]){local_size, 1, 1},
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueNDRangeKernel(queues[2], vec_mul_scalar, 1, 
		NULL, 
		(size_t[]){array_lengths[2], 1, 1},
		(size_t[]){local_size, 1, 1},
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueNDRangeKernel(queues[3], vec_sum,  1, 
		NULL, 
		(size_t[]){array_lengths[3], 1, 1},
		(size_t[]){local_size, 1, 1},
		0, NULL, NULL);
	check_error(ret);
	
	// verify results
	ret = clEnqueueReadBuffer(queues[2], buffers[2], CL_TRUE, 
		0, array_lengths[2] * sizeof(float), buf_2, 
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueReadBuffer(queues[2], buffers[3], CL_TRUE, 
		0, array_lengths[3] * sizeof(float), buf_3, 
		0, NULL, NULL);
	check_error(ret);
	
	ret = clEnqueueReadBuffer(queues[3], mem_results, CL_TRUE, 
		0, num_groups * sizeof(float), results, 
		0, NULL, NULL);
	check_error(ret);
	
	for(size_t i = 0; i < array_lengths[3]; ++i) {
		printf("[%.4d]: %g * %g = %g\n", (int)i, buf_2[i], a_2, buf_3[i]);
		assert(buf_2[i] * a_2 == buf_3[i]);
	}
	fflush(stdout);
	
	float sum = 0.0;
	for(size_t i = 0; i < num_groups; ++i) sum += results[i];
	printf("sum() = %.1f\n", sum);
	
	float sum_verify = 0.0;
	for(size_t i = 0; i < array_lengths[3]; ++i)
	{
		sum_verify += buf_3[i];
	}
	printf("sum_verify = %.1f\n", sum_verify);
	assert(sum == sum_verify);
	

// cleanup
	//release command queues
	for(int i = 0; i < NUM_COMMAND_QUEUE; ++i) {
		if(queues[i]) {
			clReleaseCommandQueue(queues[i]);
			queues[i] = NULL;
		}
	}
	
	// release kernels
	if(vec_add_scalar_0) clReleaseKernel(vec_add_scalar_0);
	if(vec_add_scalar_1) clReleaseKernel(vec_add_scalar_1);
	if(vec_mul_scalar) clReleaseKernel(vec_mul_scalar);
	if(vec_sum) clReleaseKernel(vec_sum);
	
	// release cl_mems
	for(int i = 0; i < NUM_COMMAND_QUEUE; ++i) {
		if(buffers[i]) {
			clReleaseMemObject(buffers[i]);
		}
	}
	if(mem_results) clReleaseMemObject(mem_results);
	
	// release host mems
	if(buf_0) free(buf_0);
	if(buf_1) free(buf_1);
	if(buf_2) free(buf_2);
	if(buf_3) free(buf_3);
	if(results) free(results);
	
	
	if(program) clReleaseProgram(program);
	if(sources) free(sources);
	clReleaseContext(ctx);
	return 0;
}


