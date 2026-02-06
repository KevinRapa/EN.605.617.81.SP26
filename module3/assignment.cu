//Based on the work of Andrew Krepps
#include <stdio.h>

// These next two functions are the 'simple' algorithms, similar to the ones in
// assignment.c
__global__ void gpu_kernel_no_branching(int *buffer_in, int *buffer_out)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	// blockDim.x does not depend on the thread index, so the threads in the warp should not diverge
	// (blockDim.x) / 2 is the average of the loop iterations in gpu_kernel_branching, so I think this is a comparable algorithm instruction-wise
	for (int j = 0; j < blockDim.x / 2; j++) {
		// No 'if', increments J only if buffer_out[i] is odd
		buffer_out[i] += j * (buffer_in[i] & 1);
	}
}

__global__ void gpu_kernel_branching(int *buffer_in, int *buffer_out)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	// loop iterations depend on threadIdx, so this should cause divergence
	for (int j = 0; j < threadIdx.x; j++) {
		// Increments by 'j' using an 'if' statement. Should cause divergence
		if (buffer_in[i] & 1 == 1) {
			buffer_out[i] += j;
		}
	}
}

void gpu_no_branching(int numBlocks, int blockSize, int *buffer_in, int *buffer_out)
{
	gpu_kernel_no_branching<<<numBlocks, blockSize>>>(buffer_in, buffer_out);
}

void gpu_with_branching(int numBlocks, int blockSize, int *buffer_in, int *buffer_out)
{
	gpu_kernel_branching<<<numBlocks, blockSize>>>(buffer_in, buffer_out);
}

// Times gpu kernels using events. Allocated memory and populates with ascending digits
float do_benchmark(int blockSize, int totalThreads, void (*benchmark_f)(int blockSize, int totalThreads, int *buffer_in, int *buffer_out))
{
	int *buffer_in = (int*)malloc(totalThreads * sizeof(int));
	int *buffer_out = (int*)malloc(totalThreads * sizeof(int));
	int *gpu_buf_in;
	int *gpu_buf_out;
	cudaEvent_t start;
	cudaEvent_t stop;
	int numBlocks = totalThreads / blockSize;

	// Populate with ascending digits
	for (int i = 0; i < totalThreads; i++) {
		buffer_in[i] = i;
	}

	// Alocate memory and copy into GPU
	cudaMalloc((void**)&gpu_buf_in, totalThreads * sizeof(int));
	cudaMalloc((void**)&gpu_buf_out, totalThreads * sizeof(int));

	cudaMemcpy((void *)gpu_buf_in, (void *)buffer_in, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timer and run kernel using function parameter we passed in
	cudaEventRecord(start);
	benchmark_f(numBlocks, blockSize, gpu_buf_in, gpu_buf_out);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaError_t last_error = cudaGetLastError();
	if (last_error != 0) {
		printf("CUDA error occurred %d\n", last_error);
	}

	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaMemcpy((void *)buffer_out, (void *)gpu_buf_out, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(gpu_buf_in);
	cudaFree(gpu_buf_out);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(buffer_out);
	free(buffer_in);

	return ms;
}

// Does a benchmark with both `gpu_no_branching` function and `gpu_with_branching` function
void main_sub(int blockSize, int totalThreads)
{
	printf("Running GPU benchmark functions with %d block size and %d total threads\n", blockSize, totalThreads);

	float elapsedMsNoBranching = do_benchmark(blockSize, totalThreads, gpu_no_branching);
	float elapsedMsWithBranching = do_benchmark(blockSize, totalThreads, gpu_with_branching);

	fprintf(stderr, "%s,%s,%d,%d,%f\n", "gpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranching);
	fprintf(stderr, "%s,%s,%d,%d,%f\n", "gpu", "branch", totalThreads, blockSize, elapsedMsWithBranching);
}

// Unchanged main function from template. Calls main_sub
int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	main_sub(blockSize, totalThreads);

	return 0;
}
