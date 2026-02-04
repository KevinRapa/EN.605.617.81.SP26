#include <stdio.h>

__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0(unsigned num_blocks, unsigned num_threads)
{
	printf("Running with %u blocks and %u threads\n", num_blocks, num_threads);

	const unsigned ARRAY_SIZE = num_blocks * num_threads;
	const unsigned ARRAY_SIZE_IN_BYTES = sizeof(unsigned int) * ARRAY_SIZE;

	unsigned int *cpu_block;
	unsigned int *cpu_thread;

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	cpu_block = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_thread = (unsigned *)malloc(ARRAY_SIZE_IN_BYTES);

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u\n",cpu_thread[i],cpu_block[i]);
	}

	free(cpu_block);
	free(cpu_thread);
}

int main()
{
	main_sub0(16,16);
	main_sub0(1,32);
	main_sub0(32,1);
	main_sub0(1,1);
	main_sub0(17,13);
	main_sub0(512, 512);

	return EXIT_SUCCESS;
}
