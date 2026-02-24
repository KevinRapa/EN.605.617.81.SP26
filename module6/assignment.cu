
#include <stdio.h>

__global__ void kernelGetDivisible(int *divisiblesOut, int *countOut, int *data, int divisor)
{
	extern __shared__ int count[];  // Length should equal divisor

	int threadId = threadIdx.x;

	int modulus = data[threadId] % divisor;

	int queueIdx = atomicAdd(count + modulus, 1);

	if (modulus == 0) {
		divisiblesOut[queueIdx] = data[threadId];
	}

	__syncthreads();

	*countOut = count[0];
	count[0] = 0;
}


__host__ void generate_randoms(int *out, int size, int maxValue)
{
	srand(time(NULL));

	for (int i = 0; i < size; i++) {
		out[i] = (int)rand() % maxValue;
	}
}

__host__ void printArray(int *data, int size, const char *name)
{
	printf("%s: [ ", name);

	for (int i = 0; i < size; i++) {
		printf("%d, ", data[i]);
	}

	printf(" ]\n");
}

void callKernelWithDivisor(int *divisiblesOut, int *countOut, int *dataD, int length, int divisor, int numDivisors)
{	
	int *countD = NULL;
	int *divisiblesD = NULL;

	cudaMalloc(&countD, sizeof(*countD));
	cudaMalloc(&divisiblesD, length * sizeof(*divisiblesD));

	kernelGetDivisible<<<1, length, divisor * sizeof(int)>>>(divisiblesD, countD, dataD, divisor);

	cudaDeviceSynchronize();

	cudaMemcpy(divisiblesOut, divisiblesD, length * sizeof(*divisiblesD), cudaMemcpyDeviceToHost);
	cudaMemcpy(countOut, countD, sizeof(*countD), cudaMemcpyDeviceToHost);

	cudaFree(divisiblesD);
	cudaFree(countD);
}

int main(int argc, char **argv)
{
	int *data = NULL;
	int *dataD = NULL;
	int **lists = NULL;
	int *counts = NULL;

	if (argc < 4) {
		printf("Must supply length, max value, and a list of numbers\n");
		return 1;
	}

	int arrayLength = atoi(argv[1]);
	int maxValue = atoi(argv[2]);
	int arraySize = arrayLength * sizeof(*data);
	int numDivisors = argc - 3;

	// List of lists. Each lists hold numbers divisible by the corresponding divisor
	lists = (int**)malloc(numDivisors * sizeof(*lists));

	// List of counts; length of each list in `lists`
	counts = (int*)malloc(numDivisors * sizeof(*counts));

	// For each divisor, allocate memory to store the numbers divisible by the divisor
	for (int i = 0; i < numDivisors; i++) {
		cudaMallocHost(lists + i, arraySize);
	}

	// Generate input data
	cudaMallocHost(&data, arraySize);
	cudaMalloc(&dataD, arraySize);
	generate_randoms(data, arrayLength, maxValue);
	printArray(data, arrayLength, "Data");
	cudaMemcpy(dataD, data, arraySize, cudaMemcpyHostToDevice);

	for (int i = 0; i < numDivisors; i++) {
		int currentDivisor = atoi(argv[3 + i]);
		callKernelWithDivisor(lists[i], counts + i, dataD, arrayLength, currentDivisor, numDivisors);
		printf("Number divisible by %d: %d\n", currentDivisor, counts[i]);
		printArray(lists[i], counts[i], "Divisible");
	}

	// Free memory
	cudaFreeHost(data);
	cudaFree(dataD);

	for (int i = 0; i < numDivisors; i++) {
		cudaFreeHost(lists[i]);
	}

	free(lists);
	free(counts);

	return 0;
}
