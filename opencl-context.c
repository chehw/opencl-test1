/*
 * opencl-context.c
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

#define cl_check_error(ret)  assert(CL_SUCCESS == (ret))
	

/********************************************************
 * opencl platform
********************************************************/
static inline ssize_t get_platform_string_info(cl_platform_id id, cl_platform_info key, char ** p_value)
{
	cl_int ret = 0;
	char param_buf[OPENCL_TEXT_BUFFER_SIZE] = "";
	size_t cb_param = 0;
	ret = clGetPlatformInfo(id, key, sizeof(param_buf), param_buf, &cb_param);
	cl_check_error(ret);
	
	if(NULL == p_value) return cb_param + 1;	// return buffer size (included the '\0' terminator)
	char * value = *p_value;
	if(NULL == value) {
		value = calloc(cb_param + 1, 1);
		assert(value);
		*p_value = value;
	}
	memcpy(value, param_buf, cb_param);
	value[cb_param] = '\0';
	return cb_param;
}

struct opencl_platform * opencl_platform_init(struct opencl_platform * platform, cl_platform_id id)
{
	if(NULL == platform) {
		platform = malloc(sizeof(*platform));
		assert(platform);
	}
	memset(platform, 0, sizeof(*platform));
	
	platform->id = id;
	platform->cb_profile = get_platform_string_info(id, CL_PLATFORM_PROFILE, &platform->profile);
	platform->cb_version = get_platform_string_info(id, CL_PLATFORM_VERSION, &platform->version);
	platform->cb_name = get_platform_string_info(id, CL_PLATFORM_NAME, &platform->name);
	platform->cb_vendor = get_platform_string_info(id, CL_PLATFORM_VENDOR, &platform->vendor);
	platform->cb_extensions = get_platform_string_info(id, CL_PLATFORM_EXTENSIONS, &platform->extensions);
	
#if CL_VERSION_MAJOR >= 2 && CL_VERSION_MINOR >= 1
	///< @todo ...
#endif
	return platform;
}

void opencl_platform_cleanup(struct opencl_platform * platform)
{
	if(NULL == platform) return;
	free(platform->profile);
	free(platform->version);
	free(platform->name);
	free(platform->vendor);
	free(platform->extensions);
	memset(platform, 0, sizeof(*platform));
}
void opencl_platform_dump(const struct opencl_platform * platform)
{
	assert(platform);
	fprintf(stderr, "==== %s(%p) ====\n", __FUNCTION__, platform);
	fprintf(stderr, "platform_id: %p\n", platform->id);
	fprintf(stderr, "profile: %s\n", platform->profile);
	fprintf(stderr, "version: %s\n", platform->version);
	fprintf(stderr, "name: %s\n", platform->name);
	fprintf(stderr, "vendor: %s\n", platform->vendor);
	fprintf(stderr, "extensions: %s\n", platform->extensions);
	return;
}

/********************************************************
 * opencl device
********************************************************/

void opencl_variable_clear(struct opencl_variable * var)
{
	if(NULL == var) return;
	if(var->data) {
		free(var->data);
		var->data = NULL;
	}
	var->max_size = 0;
	var->length = 0;
	return;
}

#define OPENCL_V2_1_DEVICE_INFO_LAST_ITEM 	CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS

#ifndef OPENCL_DEVICE_INFO_LAST_ITEM
#define OPENCL_DEVICE_INFO_LAST_ITEM	OPENCL_V2_1_DEVICE_INFO_LAST_ITEM
#endif

#ifndef OPENCL_DEVICE_INFO_COUNT
#define OPENCL_DEVICE_INFO_COUNT (OPENCL_DEVICE_INFO_LAST_ITEM - CL_DEVICE_TYPE + 1)
#endif

static const char * cl_device_type_to_string(unsigned int type)
{
	static char sz_type[1024] = ""; 	// todo: use thread local storage
	sz_type[0] = '\0';
	char * p = sz_type;
	char * p_end = p + sizeof(sz_type);
	
	if(type & CL_DEVICE_TYPE_CPU) p += snprintf(p, p_end - p, " (%s)", "CPU");
	if(type & CL_DEVICE_TYPE_GPU) p += snprintf(p, p_end - p, " (%s)", "GPU");
	if(type & CL_DEVICE_TYPE_ACCELERATOR) p += snprintf(p, p_end - p, " (%s)", "ACCELERATOR");
	if(type & CL_DEVICE_TYPE_CUSTOM) p += snprintf(p, p_end - p, " (%s)", "CUSTOM");
	if(type & CL_DEVICE_TYPE_DEFAULT) p += snprintf(p, p_end - p, " (%s)", "DEFAULT");
	return sz_type;
}

struct opencl_device * opencl_device_init(struct opencl_device * device, cl_device_id id)
{
	if(NULL == device) {
		device = malloc(sizeof(*device));
		assert(device);
	}
	memset(device, 0, sizeof(*device));
	
	device->id = id;
	int max_params = OPENCL_DEVICE_INFO_COUNT;
	assert(max_params > 0);
	
	cl_int ret = 0;
	struct opencl_variable * params = calloc(max_params, sizeof(*params));
	assert(params);
	
	device->params = params;
	device->num_params = max_params;
	
	for(int i = 0; i < max_params; ++i) {
		struct opencl_variable * param = &params[i];
		int index = CL_DEVICE_TYPE + i;
		
		char buf[PATH_MAX] = "";
		size_t cb_param = 0;
		ret = clGetDeviceInfo(id, index, sizeof(buf), buf, &cb_param);
		if(ret == CL_SUCCESS) {
			param->data_type = opencl_data_type_unparsed;
			
			// todo: set the corresponding data type
			// currently only copying the binary data without any parsing 
			param->max_size = cb_param + 1;
			param->length = cb_param;
			
			void * data = calloc(cb_param + 1, 1);
			memcpy(data, buf, cb_param);
			param->data = data;
		} else {
			fprintf(stderr, "\e[33m[WARNING]: device info (%d)(0x%.8x) not found\e[39m\n", index, index);
		}
	}
	
	// parse device_type
	struct opencl_variable * param = &params[CL_DEVICE_TYPE - CL_DEVICE_TYPE];
	assert(param->data);	
	device->device_type = *(cl_device_type *)param->data;
	
	// parse max_compute_units
	param = &params[CL_DEVICE_MAX_COMPUTE_UNITS - CL_DEVICE_TYPE];
	assert(param->data);
	device->max_compute_units = *(cl_uint *)param->data;
	
	// parse max_work_item_dimensions
	param = &params[CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS - CL_DEVICE_TYPE];
	assert(param->data);
	device->max_work_item_demensions = *(cl_uint *)param->data;
	
	// parse max_work_group_size
	param = &params[CL_DEVICE_MAX_WORK_GROUP_SIZE - CL_DEVICE_TYPE];
	assert(param->data);	
	device->max_work_group_size = *(cl_uint *)param->data;
	
	// parse max_mem_alloc_size
	param = &params[CL_DEVICE_MAX_MEM_ALLOC_SIZE - CL_DEVICE_TYPE];
	assert(param->data);
	device->max_mem_alloc_size = *(cl_ulong *)param->data;
	
	// parse is_availabable
	param = &params[CL_DEVICE_AVAILABLE - CL_DEVICE_TYPE];
	assert(param->data);
	device->is_available = (CL_FALSE != *(cl_bool *)param->data);
	
	// parse max_sub_devices
	param = &params[CL_DEVICE_PARTITION_MAX_SUB_DEVICES - CL_DEVICE_TYPE];
	assert(param->data);
	device->max_sub_devices = *(cl_uint *)param->data;
	
	return device;
}

void opencl_device_cleanup(struct opencl_device * device)
{
	if(NULL == device) return;
	if(device->params) {
		for(int i = 0; i < device->num_params; ++i) {
			struct opencl_variable * param = &device->params[i];
			opencl_variable_clear(param);
		}
		free(device->params);
		device->params = NULL;
	}
	device->num_params = 0;
	return;
}

void opencl_device_dump(const struct opencl_device * device)
{
	assert(device);
	fprintf(stderr, "==== %s(%p) ====\n", __FUNCTION__, device);
	fprintf(stderr, "  device_type: %s\n", cl_device_type_to_string(device->device_type));
	fprintf(stderr, "  max_compute_units: %u\n", (unsigned int)device->max_compute_units);
	fprintf(stderr, "  max_work_item_demensions: %u\n", (unsigned int)device->max_work_item_demensions);
	fprintf(stderr, "  max_work_group_size: %lu\n", (unsigned long)device->max_work_group_size);
	fprintf(stderr, "  max_mem_alloc_size: %lu\n", (unsigned long)device->max_mem_alloc_size);
	fprintf(stderr, "  is_available: %s\n", device->is_available?"True":"False");
	fprintf(stderr, "  max_sub_devices: %u\n", (unsigned int)device->max_sub_devices);
	
	fprintf(stderr, "  -- dump all params(num_params=%d) --\n", (int)device->num_params);
	if(device->params) {
		for(int i = 0; i < device->num_params; ++i) {
			struct opencl_variable * param = &device->params[i];
			if(NULL == param->data) continue;
			
			int device_info_index = CL_DEVICE_TYPE + i;
			switch(device_info_index) 
			{
			case CL_DEVICE_TYPE:
				fprintf(stderr, "\t" "DEVICE_TYPE: %s\n", cl_device_type_to_string(*(cl_device_type *)param->data));
				break;
			case CL_DEVICE_VENDOR_ID:
				fprintf(stderr, "\t" "VENDOR_ID: %u\n", *(uint32_t *)param->data);
				break;
			case CL_DEVICE_MAX_COMPUTE_UNITS:
				fprintf(stderr, "\t" "MAX_COMPUTE_UNITS: %u\n", *(uint32_t *)param->data);
				break;
			case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
				fprintf(stderr, "\t" "MAX_WORK_ITEM_DIMENSIONS: %u\n", *(uint32_t *)param->data);
				break;
			case CL_DEVICE_MAX_WORK_GROUP_SIZE:
				fprintf(stderr, "\t" "MAX_WORK_GROUP_SIZE: %lu\n", (unsigned long)*(size_t *)param->data);
				break;
			case CL_DEVICE_MAX_WORK_ITEM_SIZES:
				fprintf(stderr, "\t" "MAX_WORK_ITEM_SIZES: [");
				{
					size_t array_length = param->length / sizeof(size_t);
					assert(array_length == (size_t)device->max_work_item_demensions);
					
					size_t * sizes = (size_t *)param->data;
					for(size_t ii = 0; ii < array_length; ++ii) {
						if(ii > 0) fprintf(stderr, ", ");
						fprintf(stderr, "%lu", (unsigned long)sizes[ii]);
					}
					fprintf(stderr, "]\n");
				}
				break;
				
			
			///< @todo ...
			// case ...
			case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
				fprintf(stderr, "\t" "MAX_MEM_ALLOC_SIZE: %lu\n", (unsigned long)*(cl_ulong *)param->data);
				break;
			
			case CL_DEVICE_GLOBAL_MEM_SIZE:
				fprintf(stderr, "\t" "CLOBAL_MEM_SIZE: %lu\n", *(unsigned long *)param->data);
				break;
			
			///< @todo ...
			// case ...
			
			case CL_DEVICE_NAME:
				fprintf(stderr, "\t" "CL_DEVICE_NAME: %s\n", (char *)param->data);
				break;
			case CL_DEVICE_VENDOR:
				fprintf(stderr, "\t" "CL_DEVICE_VENDOR: %s\n", (char *)param->data);
				break;
			case CL_DRIVER_VERSION:
				fprintf(stderr, "\t" "CL_DRIVER_VERSION: %s\n", (char *)param->data);
				break;
			case CL_DEVICE_PROFILE:
				fprintf(stderr, "\t" "CL_DEVICE_PROFILE: %s\n", (char *)param->data);
				break;
			case CL_DEVICE_VERSION:
				fprintf(stderr, "\t" "CL_DEVICE_VERSION: %s\n", (char *)param->data);
				break;
			case CL_DEVICE_OPENCL_C_VERSION:
				fprintf(stderr, "\t" "CL_DEVICE_OPENCL_C_VERSION: %s\n", (char *)param->data);
				break;
			case CL_DEVICE_EXTENSIONS:
				fprintf(stderr, "\t" "CL_DEVICE_EXTENSIONS: %s\n", (char *)param->data);
				break;
				
			// todo: ...
			default:
				if(param->data) {
					fprintf(stderr, "\t" "device_info_(0x%x): (length=%ld) u32=%u(%.8x)(str=%*s)\n", 
						device_info_index, 
						(long)param->length,
						*(uint32_t *)param->data,
						*(uint32_t *)param->data,
						(int)param->length, (char *)param->data
					);
				}
				break;
			}
		#if defined(WIN32) || defined(_WIN32)
			/* 
			 * When testing opencl-lib under msys2 (intel-uhd + nvidia-cuda 11.3, win10_x64),
			 * puts/printf() outputs nothing to the terminal(mintty-3.4.7)  unless fflush() is called explicitly.
			*/
			fflush(stderr);
		#endif
		}
	}
}

/********************************************************
 * opencl context
********************************************************/
static struct opencl_platform * get_platform_by_name_prefix(struct opencl_context * cl, const char * name)
{
	if(cl->num_platforms <= 0) return NULL;
	if(NULL == name) return &cl->platforms[0];
	int cb_name = strlen(name);
	if(cb_name <= 0) return NULL;
	
	for(int i = 0; i < cl->num_platforms; ++i) {
		struct opencl_platform * platform = &cl->platforms[i];
		assert(platform && platform->name);
		if(strncasecmp(platform->name, name, cb_name) == 0) return platform;
	}
	return NULL;
}

static int load_devices(struct opencl_context * cl, cl_device_type device_type, const struct opencl_platform * platform)
{
	assert(cl);
	if(NULL == platform || NULL == platform->id) return -1;
	
	cl_uint available_devices = 0;
	cl_uint num_devices = 0;
	
	// clear current devices
	if(cl->devices) {
		for(int i = 0; i < cl->num_devices; ++i) {
			struct opencl_device * device = &cl->devices[i];
			opencl_device_cleanup(device);
		}
		free(cl->devices);
		cl->devices = NULL;
	}
	cl->num_devices = 0;
	
	if(0 == device_type) device_type = CL_DEVICE_TYPE_ALL;
	
	cl_int ret = clGetDeviceIDs(platform->id, device_type, 0, NULL, &available_devices);
	cl_check_error(ret);
	assert(available_devices > 0);
	
	cl_device_id * device_ids = calloc(available_devices, sizeof(*device_ids));
	assert(device_ids);
	
	ret = clGetDeviceIDs(platform->id, device_type, available_devices, device_ids, &num_devices);
	cl_check_error(ret);
	assert(num_devices > 0 && num_devices <= available_devices);
	
	struct opencl_device * devices = calloc(num_devices, sizeof(*devices));
	assert(devices);
	
	fprintf(stderr, "[INFO]: num_devices: %d\n", (int)num_devices);
	for(int i = 0; i < num_devices; ++i) {
		struct opencl_device * device = opencl_device_init(&devices[i], device_ids[i]);
		opencl_device_dump(device);
	}
	cl->num_devices = num_devices;
	cl->devices = devices;
	
	return 0;
}

opencl_context_t * opencl_context_init(opencl_context_t * cl, void * user_data)
{
	if(NULL == cl) cl = calloc(1, sizeof(*cl));
	assert(cl);
	
	cl->user_data = user_data;
	cl->load_devices = load_devices;
	cl->get_platform_by_name_prefix = get_platform_by_name_prefix;
	
	cl_platform_id * platform_ids = NULL;
	cl_uint num_available = 0;
	cl_uint num_platforms = 0;
	
	cl_int ret = clGetPlatformIDs(0, NULL, &num_available);
	cl_check_error(ret);
	
	printf("avaliable platforms: %d\r\n", num_available);
	platform_ids = calloc(num_available, sizeof(*platform_ids));
	assert(platform_ids);
	
	ret = clGetPlatformIDs(num_available, platform_ids, &num_platforms);
	cl_check_error(ret);
	assert(num_platforms <= num_available);
	
	struct opencl_platform * platforms = calloc(num_platforms, sizeof(*platforms));
	assert(platforms);
	
	cl->num_platforms = num_platforms;
	cl->platforms = platforms;
	
	for(cl_uint i = 0; i < num_platforms; ++i)
	{
		struct opencl_platform * platform = opencl_platform_init(&platforms[i],  platform_ids[i]);
		assert(platform);
		
		opencl_platform_dump(platform);
	}
	free(platform_ids);
	return cl;
	
}

void opencl_context_cleanup(opencl_context_t * cl)
{
	if(NULL == cl) return;
	for(int i = 0; i < cl->num_platforms; ++i)
	{
		opencl_platform_cleanup(&cl->platforms[i]);
	}
	return;
}




/**
 *  utils
**/

#define CASE_RETURN_STR(enum_index) case enum_index: return #enum_index;
 
const char * opencl_error_to_string(cl_int err_code)
{
	switch(err_code) {
	CASE_RETURN_STR(CL_SUCCESS)
	CASE_RETURN_STR(CL_DEVICE_NOT_FOUND)
	CASE_RETURN_STR(CL_DEVICE_NOT_AVAILABLE)
	CASE_RETURN_STR(CL_COMPILER_NOT_AVAILABLE)
	CASE_RETURN_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE)
	CASE_RETURN_STR(CL_OUT_OF_RESOURCES)
	CASE_RETURN_STR(CL_OUT_OF_HOST_MEMORY)
	CASE_RETURN_STR(CL_PROFILING_INFO_NOT_AVAILABLE)
	CASE_RETURN_STR(CL_MEM_COPY_OVERLAP)
	CASE_RETURN_STR(CL_IMAGE_FORMAT_MISMATCH)
	CASE_RETURN_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED)
	CASE_RETURN_STR(CL_BUILD_PROGRAM_FAILURE)
	CASE_RETURN_STR(CL_MAP_FAILURE)
	CASE_RETURN_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET)
	CASE_RETURN_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
	CASE_RETURN_STR(CL_COMPILE_PROGRAM_FAILURE)
	CASE_RETURN_STR(CL_LINKER_NOT_AVAILABLE)
	CASE_RETURN_STR(CL_LINK_PROGRAM_FAILURE)
	CASE_RETURN_STR(CL_DEVICE_PARTITION_FAILED)
	CASE_RETURN_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
	CASE_RETURN_STR(CL_INVALID_VALUE)
	CASE_RETURN_STR(CL_INVALID_DEVICE_TYPE)
	CASE_RETURN_STR(CL_INVALID_PLATFORM)
	CASE_RETURN_STR(CL_INVALID_DEVICE)
	CASE_RETURN_STR(CL_INVALID_CONTEXT)
	CASE_RETURN_STR(CL_INVALID_QUEUE_PROPERTIES)
	CASE_RETURN_STR(CL_INVALID_COMMAND_QUEUE)
	CASE_RETURN_STR(CL_INVALID_HOST_PTR)
	CASE_RETURN_STR(CL_INVALID_MEM_OBJECT)
	CASE_RETURN_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
	CASE_RETURN_STR(CL_INVALID_IMAGE_SIZE)
	CASE_RETURN_STR(CL_INVALID_SAMPLER)
	CASE_RETURN_STR(CL_INVALID_BINARY)
	CASE_RETURN_STR(CL_INVALID_BUILD_OPTIONS)
	CASE_RETURN_STR(CL_INVALID_PROGRAM)
	CASE_RETURN_STR(CL_INVALID_PROGRAM_EXECUTABLE)
	CASE_RETURN_STR(CL_INVALID_KERNEL_NAME)
	CASE_RETURN_STR(CL_INVALID_KERNEL_DEFINITION)
	CASE_RETURN_STR(CL_INVALID_KERNEL)
	CASE_RETURN_STR(CL_INVALID_ARG_INDEX)
	CASE_RETURN_STR(CL_INVALID_ARG_VALUE)
	CASE_RETURN_STR(CL_INVALID_ARG_SIZE)
	CASE_RETURN_STR(CL_INVALID_KERNEL_ARGS)
	CASE_RETURN_STR(CL_INVALID_WORK_DIMENSION)
	CASE_RETURN_STR(CL_INVALID_WORK_GROUP_SIZE)
	CASE_RETURN_STR(CL_INVALID_WORK_ITEM_SIZE)
	CASE_RETURN_STR(CL_INVALID_GLOBAL_OFFSET)
	CASE_RETURN_STR(CL_INVALID_EVENT_WAIT_LIST)
	CASE_RETURN_STR(CL_INVALID_EVENT)
	CASE_RETURN_STR(CL_INVALID_OPERATION)
	CASE_RETURN_STR(CL_INVALID_GL_OBJECT)
	CASE_RETURN_STR(CL_INVALID_BUFFER_SIZE)
	CASE_RETURN_STR(CL_INVALID_MIP_LEVEL)
	CASE_RETURN_STR(CL_INVALID_GLOBAL_WORK_SIZE)
	CASE_RETURN_STR(CL_INVALID_PROPERTY)
	CASE_RETURN_STR(CL_INVALID_IMAGE_DESCRIPTOR)
	CASE_RETURN_STR(CL_INVALID_COMPILER_OPTIONS)
	CASE_RETURN_STR(CL_INVALID_LINKER_OPTIONS)
	CASE_RETURN_STR(CL_INVALID_DEVICE_PARTITION_COUNT)
	CASE_RETURN_STR(CL_INVALID_PIPE_SIZE)
	CASE_RETURN_STR(CL_INVALID_DEVICE_QUEUE)
	CASE_RETURN_STR(CL_INVALID_SPEC_ID)
	CASE_RETURN_STR(CL_MAX_SIZE_RESTRICTION_EXCEEDED)
	default:
		break;
	}
	return "unknown error";
}
