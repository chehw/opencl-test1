#ifndef OPENCL_CONTEXT_H_
#define OPENCL_CONTEXT_H_

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

#define OPENCL_TEXT_BUFFER_SIZE (4096)
/**
 * opencl platform
 */
struct opencl_platform
{
	cl_platform_id id;
	char * profile;
	size_t cb_profile;
	
	char * version;
	size_t cb_version;
	
	char * name;
	size_t cb_name;
	
	char * vendor;
	size_t cb_vendor;
	
	char * extensions;
	size_t cb_extensions;
};
struct opencl_platform * opencl_platform_init(struct opencl_platform * platform, cl_platform_id id);
void opencl_platform_cleanup(struct opencl_platform * platform);
void opencl_platform_dump(const struct opencl_platform * platform);


enum opencl_data_type
{
	opencl_data_type_unparsed,
	opencl_type_char_array,
	opencl_type_int,
	opencl_type_uint,
	opencl_type_ulong,
	opencl_type_bool,
	opencl_type_size_t,
	opencl_type_size_array,
	opencl_type_device_fp_config,
	opencl_type_enum_type,
	opencl_type_enums_array,
};

struct opencl_variable
{
	enum opencl_data_type data_type;
	size_t max_size;
	size_t length;
	void * data;
};
void opencl_variable_clear(struct opencl_variable * var);
char * opencl_variable_to_string(const struct opencl_variable * var);


/**
 * opencl device
 */
struct opencl_device
{
	cl_device_id id;
	cl_device_type device_type;			// CL_DEVICE_TYPE                                   0x1000
	const struct opencl_platform * platform;
	
	cl_uint max_compute_units;			// CL_DEVICE_MAX_COMPUTE_UNITS                      0x1002
	cl_uint max_work_item_demensions;	// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS               0x1003
	size_t max_work_group_size;			// CL_DEVICE_MAX_WORK_GROUP_SIZE                    0x1004
	size_t max_mem_alloc_size;			// CL_DEVICE_MAX_MEM_ALLOC_SIZE                     0x1010
	int is_available;					// CL_DEVICE_AVAILABLE                              0x1027
	cl_uint max_sub_devices;			// CL_DEVICE_PARTITION_MAX_SUB_DEVICES              0x1043
	// ...
	
	int num_params;
	struct opencl_variable *params;
};
struct opencl_device * opencl_device_init(struct opencl_device * device, cl_device_id id);
void opencl_device_cleanup(struct opencl_device * device);
void opencl_device_dump(const struct opencl_device * device);


/**
 * opencl context
 */

typedef struct opencl_context
{
	void * priv;
	void * user_data;
	
	int num_platforms;
	struct opencl_platform * platforms;
	
	int num_devices;
	struct opencl_device * devices;
	
	int (* load_devices)(struct opencl_context * cl, cl_device_type device_type, const struct opencl_platform * platform);
	struct opencl_platform * (*get_platform_by_name_prefix)(struct opencl_context * cl, const char * platform_name_prefix);

}opencl_context_t;
opencl_context_t * opencl_context_init(opencl_context_t * cl, void * user_data);
void opencl_context_cleanup(opencl_context_t * cl);


extern const char * opencl_error_to_string(cl_int err_code);

#ifdef __cplusplus
}
#endif
#endif
