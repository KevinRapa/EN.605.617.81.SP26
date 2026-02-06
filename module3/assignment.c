//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

// These next two functions are the 'simple' algorithms
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

// Times the cpu functions using std::chrono
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

	fprintf(stderr, "%s,%s,%d,%d,%ld\n", "cpu", "no_branch", totalThreads, blockSize, elapsedMsNoBranching);
	fprintf(stderr, "%s,%s,%d,%d,%ld\n", "cpu", "branch", totalThreads, blockSize, elapsedMsWithBranching);
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
