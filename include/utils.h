#ifndef OPENCL_TEST_UTILS_H_
#define OPENCL_TEST_UTILS_H_

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

ssize_t load_file(const char * filename, char ** p_data);

#ifdef __cplusplus
}
#endif
#endif
