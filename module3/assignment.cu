//Based on the work of Andrew Krepps
#include <stdio.h>

__global__ void gpu_kernel_no_branching(int *buffer_in, int *buffer_out)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	buffer_out[i] = buffer_in[i] & 1;
}

__global__ void gpu_kernel_branching(int *buffer_in, int *buffer_out)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (buffer_in[i] & 1 == 1) {
		buffer_out[i] = 1;
	} else {
		buffer_out[i] = 0;
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

float do_benchmark(int blockSize, int totalThreads, void (*benchmark_f)(int blockSize, int totalThreads, int *buffer_in, int *buffer_out))
{
	int *buffer_in = (int*)malloc(totalThreads * sizeof(int));
	int *buffer_out = (int*)malloc(totalThreads * sizeof(int));
	int *gpu_buf_in;
	int *gpu_buf_out;
	cudaEvent_t start;
	cudaEvent_t stop;
	int numBlocks = totalThreads / blockSize;

	for (int i = 0; i < totalThreads; i++) {
		buffer_in[i] = i;
	}

	cudaMalloc((void**)&gpu_buf_in, totalThreads * sizeof(int));
	cudaMalloc((void**)&gpu_buf_out, totalThreads * sizeof(int));

	cudaMemcpy((void *)gpu_buf_in, (void *)buffer_in, totalThreads * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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

void main_sub(int blockSize, int totalThreads)
{
	printf("Running GPU benchmark functions with %d block size and %d total threads\n", blockSize, totalThreads);

	float elapsedMsNoBranching = do_benchmark(blockSize, totalThreads, gpu_no_branching);
	float elapsedMsWithBranching = do_benchmark(blockSize, totalThreads, gpu_with_branching);

	printf("%s,%s,%d,%d,%f\n", "gpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranching);
	printf("%s,%s,%d,%d,%f\n", "gpu", "branch", totalThreads, blockSize, elapsedMsWithBranching);
}

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
