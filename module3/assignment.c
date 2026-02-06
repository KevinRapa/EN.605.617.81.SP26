//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

// "Is odd" function
void cpu_no_branching(int *buffer_in, int *buffer_out, int size)
{
	for (int i = 0; i < size; i++) {
		buffer_out[i] = buffer_in[i] & 1;
	}
}

// "Is odd" function
void cpu_with_branching(int *buffer_in, int *buffer_out, int size)
{
	for (int i = 0; i < size; i++) {
		if (buffer_in[i] & 1 == 1) {
			buffer_out[i] = 1;
		} else {
			buffer_out[i] = 0;
		}
	}
}

long do_benchmark(int blockSize, int totalThreads, void (*benchmark_f)(int *buffer_in, int *buffer_out, int size))
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

void main_sub(int blockSize, int totalThreads)
{
	printf("Running CPU benchmark functions with %d block size and %d total threads\n", blockSize, totalThreads);

	long elapsedMsNoBranching = do_benchmark(blockSize, totalThreads, cpu_no_branching);
	long elapsedMsWithBranching = do_benchmark(blockSize, totalThreads, cpu_with_branching);

	printf("%s,%s,%d,%d,%ld\n", "cpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranching);
	printf("%s,%s,%d,%d,%ld\n", "cpu", "branch", totalThreads, blockSize, elapsedMsWithBranching);
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
