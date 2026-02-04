// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

__global__ 
void hello(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = threadIdx.x;
}

void do_hello_custom_dims(unsigned block_size, unsigned number_blocks)
{
	printf("Running hello with %u blocks of size %u\n", number_blocks, block_size);

	int *gpu_block;
	int *cpu_block;

	const unsigned NUM_THREADS = block_size * number_blocks;
	const unsigned ARRAY_SIZE_BYTES = NUM_THREADS * sizeof(*gpu_block);

	cpu_block = (int *)malloc(ARRAY_SIZE_BYTES);

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_BYTES);
	cudaMemcpy(gpu_block, cpu_block, ARRAY_SIZE_BYTES, cudaMemcpyHostToDevice);

	hello<<<number_blocks, block_size>>>(gpu_block);

	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_block);

	for(unsigned int i = 0; i < NUM_THREADS; i++) {
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}

	free(cpu_block);
}

int main()
{
	do_hello_custom_dims(16, 1);
	do_hello_custom_dims(32, 2);
	do_hello_custom_dims(256, 256);
	do_hello_custom_dims(13, 3);
	do_hello_custom_dims(33, 1);

	return EXIT_SUCCESS;
}
