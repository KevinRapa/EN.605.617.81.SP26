#include <stdio.h>

__global__ void what_is_my_id_2d_A(
				unsigned int * const block_x,
				unsigned int * const block_y,
				unsigned int * const thread,
				unsigned int * const calc_thread,
				unsigned int * const x_thread,
				unsigned int * const y_thread,
				unsigned int * const grid_dimx,
				unsigned int * const block_dimx,
				unsigned int * const grid_dimy,
				unsigned int * const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread[thread_idx] = threadIdx.x;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

unsigned index_helper(unsigned *array_2d, unsigned x_dim, unsigned x, unsigned y)
{
	return array_2d[(y * x_dim) + x];
}

void custom_dims(dim3 grid_dim, dim3 block_dim)
{
	unsigned int * gpu_block_x;
	unsigned int * gpu_block_y;
	unsigned int * gpu_thread;
	unsigned int * gpu_calc_thread;
	unsigned int * gpu_xthread;
	unsigned int * gpu_ythread;
	unsigned int * gpu_grid_dimx;
	unsigned int * gpu_block_dimx;
	unsigned int * gpu_grid_dimy;
	unsigned int * gpu_block_dimy;

	const unsigned ARRAY_SIZE_X = grid_dim.x * block_dim.x;
	const unsigned ARRAY_SIZE_Y = grid_dim.y * block_dim.y;
	const unsigned ARRAY_SIZE_IN_BYTES = ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(*gpu_block_x)));

	unsigned int *cpu_block_x;
	unsigned int *cpu_block_y;
	unsigned int *cpu_thread;
	unsigned int *cpu_calc_thread;
	unsigned int *cpu_xthread;
	unsigned int *cpu_ythread;
	unsigned int *cpu_grid_dimx;
	unsigned int *cpu_block_dimx;
	unsigned int *cpu_grid_dimy;
	unsigned int *cpu_block_dimy;

	cpu_block_x = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_block_y = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_thread = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_calc_thread = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_xthread = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_ythread = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_grid_dimx = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_block_dimx = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_grid_dimy = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_block_dimy = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);

	cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_xthread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_ythread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);

	what_is_my_id_2d_A<<<grid_dim, block_dim>>>(
		gpu_block_x,
		gpu_block_y,
		gpu_thread,
		gpu_calc_thread,
		gpu_xthread,
		gpu_ythread,
		gpu_grid_dimx,
		gpu_block_dimx,
		gpu_grid_dimy,
		gpu_block_dimy
	);

	/* Copy back the gpu results to the CPU */
	cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	printf("\nGrid(%d,%d,%d) Block(%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
	/* Iterate through the arrays and print */
	for(int y = 0; y < ARRAY_SIZE_Y; y++)
	{
		for(int x = 0; x < ARRAY_SIZE_X; x++)
		{
			printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
			       index_helper(cpu_calc_thread, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_block_x, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_block_y, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_thread, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_ythread, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_xthread, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_grid_dimx, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_block_dimx, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_grid_dimy, ARRAY_SIZE_X, x, y),
			       index_helper(cpu_block_dimy, ARRAY_SIZE_X, x, y)
			);

		}
	}

	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_xthread);
	cudaFree(gpu_ythread);
	cudaFree(gpu_grid_dimx);
	cudaFree(gpu_block_dimx);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);

	free(cpu_block_x);
	free(cpu_block_y);
	free(cpu_thread);
	free(cpu_calc_thread);
	free(cpu_xthread);
	free(cpu_ythread);
	free(cpu_grid_dimx);
	free(cpu_block_dimx);
	free(cpu_grid_dimy);
	free(cpu_block_dimy);
}

int main(void)
{
	/* Total thread count = 32 * 4 = 128 */
	const dim3 threads_rect(32,4);
	const dim3 blocks_rect(1,4);
	custom_dims(blocks_rect, threads_rect);

	/* Total thread count = 16 * 8 = 128 */
	const dim3 threads_square(16, 8);
	const dim3 blocks_square(2,2);
	custom_dims(blocks_square, threads_square);

	// 1 thread per block, 32*32 blocks total
	const dim3 threads_dim_custom1(1,1);
	const dim3 blocks_dim_custom1(32,32);
	custom_dims(blocks_dim_custom1, threads_dim_custom1);

	/* 1 block with 32*32 threads */
	const dim3 threads_dim_custom2(32,32);
	const dim3 blocks_dim_custom2(1,1);
	custom_dims(blocks_dim_custom2, threads_dim_custom2);

	/* Awkard block size (not multiple of 32) */
	const dim3 threads_dim_custom3(53, 1);
	const dim3 blocks_dim_custom3(4, 4);
	custom_dims(blocks_dim_custom3, threads_dim_custom3);

	/* Thread size 0 */
	const dim3 threads_dim_custom4(0,0);
	const dim3 blocks_dim_custom4(4, 4);
	custom_dims(blocks_dim_custom4, threads_dim_custom4);

	/* Single dimension array*/
	const dim3 threads_dim_custom5(16);
	const dim3 blocks_dim_custom5(16);
	custom_dims(blocks_dim_custom5, threads_dim_custom5);

	return 0;
}
