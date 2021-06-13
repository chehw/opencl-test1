/*
 * utils.c
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

#include "utils.h"

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


