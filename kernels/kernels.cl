/*
 * kernels.cl
 * 
 * Copyright 2021 chehw <hongwei.che@gmail.com>
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


/*
 * mul_scalar():  Y[i] = X[i] * a
 */
__kernel void vec_mul_scalar(__global float * Y, __global float * X, __const float a)
{
	const int i = get_global_id(0);
	Y[i] = X[i] * a;
	return;
}

/*
 * add_scalar():  Y[i] = X[i] + a
 */
__kernel void vec_add_scalar(__global float * Y, __global const float * X, __const float a, __const int y_offset)
{
	const int i = get_global_id(0);
	Y[i + y_offset] = X[i] + a;
	return;
}

/*
 * reduction for the vec_sum operation
 * 
 * result = sum(A)
 */
__kernel void vec_sum(__const int n, __global float * A, __local float * partials, __global float * result)
{
	int global_index = get_global_id(0);
	int local_index = get_local_id(0);
	float sum = 0.0f;
	
	if(global_index < n) {
		partials[local_index] = A[global_index];
	}else {
		partials[local_index] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	size_t block_size = get_local_size(0);
	size_t half_block_size = block_size	/ 2;
	
	while(half_block_size > 0) {
		if(local_index < half_block_size) {
			partials[local_index] += partials[local_index + half_block_size];
			if(block_size & 1) // odd number 
			{
				if(0 == local_index) partials[local_index] += partials[local_index + (block_size - 1)];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		block_size = half_block_size;
		half_block_size /= 2;
	}
	
	if(local_index == 0) {
		result[get_group_id(0)] = partials[0];
	}
}
