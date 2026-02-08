//Based on the work of Andrew Krepps
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <iostream>

// Imposing a max limit on the input array for CPU functions, since otherwise they will take too long
#define MAX_ARRAY_SIZE_CPU 100000

// These next four functions are the 'simple' algorithms used to benchmark
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

void cpu_with_branching(int *buffer_in, int *buffer_out, int size)
{
	// For every digit in buffer_in
	for (int i = 0; i < size; i++) {
		// For every number j from 0 to that digit
		for (int j = 0; j < buffer_in[i]; j++) {
			// Add j only if buffer_in[i] is odd
			if (buffer_in[i] & 1 == 1) {
				buffer_out[i] += j;
			}
		}
	}
}

// buffer_in will hold ascending digits from 0 to size-1
void cpu_no_branching(int *buffer_in, int *buffer_out, int size)
{
	// For every digit in buffer_in
	for (int i = 0; i < size; i++) {
		// For every number j from 0 to size/2
		// size/2 is average of the number of iterations for the other function
		for (int j = 0; j < size / 2; j++) {
			// Add j only if buffer_in[i] is odd
			buffer_out[i] += j * (buffer_in[i] & 1);
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

// Times the cpu functions using std::chrono
long do_benchmark_cpu(int blockSize, int totalThreads, void (*benchmark_f)(int *buffer_in, int *buffer_out, int size))
{
	(void)blockSize;  // No GPU, so we can't make use of this

	int *buffer_in = (int*)malloc(totalThreads * sizeof(int));
	int *buffer_out = (int*)malloc(totalThreads * sizeof(int));

	for (int i = 0; i < totalThreads; i++) {
		buffer_in[i] = i;
	}

	auto start = std::chrono::high_resolution_clock::now();

	benchmark_f(buffer_in, buffer_out, totalThreads);

	auto stop = std::chrono::high_resolution_clock::now();

	free(buffer_out);
	free(buffer_in);

	return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}
// Times gpu kernels using events. Allocated memory and populates with ascending digits
float do_benchmark_gpu(int blockSize, int totalThreads, void (*benchmark_f)(int blockSize, int totalThreads, int *buffer_in, int *buffer_out))
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

// Runs the benchmarking functions
void main_sub(int blockSize, int totalThreads)
{
	printf("Running benchmark functions with %d block size and %d total threads\n\n", blockSize, totalThreads);

	// Pipe to stderr to collect results easier
	fprintf(stderr, "%s,%s,%s,%s,%s\n", "type", "behavior", "threadCount", "blockSize", "millisecondsElapsed");

	float elapsedMsNoBranchingGPU = do_benchmark_gpu(blockSize, totalThreads, gpu_no_branching);
	float elapsedMsWithBranchingGPU = do_benchmark_gpu(blockSize, totalThreads, gpu_with_branching);

	fprintf(stderr, "%s,%s,%d,%d,%f\n", "gpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranchingGPU);
	fprintf(stderr, "%s,%s,%d,%d,%f\n", "gpu", "branch", totalThreads, blockSize, elapsedMsWithBranchingGPU);

	if (totalThreads < MAX_ARRAY_SIZE_CPU) {
		long elapsedMsNoBranchingCPU = do_benchmark_cpu(blockSize, totalThreads, cpu_no_branching);
		long elapsedMsWithBranchingCPU = do_benchmark_cpu(blockSize, totalThreads, cpu_with_branching);

		fprintf(stderr, "%s,%s,%d,%d,%ld\n", "cpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranchingCPU);
		fprintf(stderr, "%s,%s,%d,%d,%ld\n", "cpu", "branch", totalThreads, blockSize, elapsedMsWithBranchingCPU);
	} else {
		printf("Skipping CPU benchmark because input is too large\n");
	}
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
