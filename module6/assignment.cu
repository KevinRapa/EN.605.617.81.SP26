
#include <stdio.h>

__global__ 
__global__ void kernelSplitEvensOdds(int *evensOut, int *oddsOut, int *data, int *evensCount, int *oddsCount)
{
	__shared__ int evensOddsCount[2];

	int *evensOddsPtrs[2] = { evensOut, oddsOut };

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	bool isOdd = data[threadId] & 1;

	int queueIdx = atomicAdd(evensOddsCount + isOdd, 1);

	evensOddsPtrs[isOdd][queueIdx] = data[threadId];

	__syncthreads();

	if (threadId == 0) {
		*evensCount = evensOddsCount[0];
		*oddsCount = evensOddsCount[1];
	}
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

int main(int argc, char **argv)
{
	int *data = NULL, *evens = NULL, *odds = NULL;
	int evensCount = 0, oddsCount = 0;
	int *dataD = NULL, *evensD = NULL, *oddsD = NULL;
	int *evensCountD = NULL, *oddsCountD = NULL;


	if (argc < 3) {
		printf("Must supply length and max value\n");
		return 1;
	}

	int arrayLength = atoi(argv[1]);
	int maxValue = atoi(argv[2]);
	int arraySize = arrayLength * sizeof(*data);

	cudaMallocHost(&data, arraySize);
	cudaMallocHost(&evens, arraySize);
	cudaMallocHost(&odds, arraySize);

	cudaMalloc(&dataD, arraySize);
	cudaMalloc(&evensD, arraySize);
	cudaMalloc(&oddsD, arraySize);
	cudaMalloc(&evensCountD, sizeof(*evensCountD));
	cudaMalloc(&oddsCountD, sizeof(*oddsCountD));
	
	generate_randoms(data, arrayLength, maxValue);

	printArray(data, arrayLength, "Data");

	cudaMemcpy(dataD, data, arraySize, cudaMemcpyHostToDevice);

	kernelSplitEvensOdds<<<1, arrayLength>>>(evensD, oddsD, dataD, evensCountD, oddsCountD);

	cudaMemcpy(&evensCount, evensCountD, sizeof(*evensCountD), cudaMemcpyDeviceToHost);
	cudaMemcpy(&oddsCount, oddsCountD, sizeof(*oddsCountD), cudaMemcpyDeviceToHost);
	cudaMemcpy(evens, evensD, evensCount * sizeof(*evensD), cudaMemcpyDeviceToHost);
	cudaMemcpy(odds, oddsD, oddsCount * sizeof(*oddsD), cudaMemcpyDeviceToHost);

	cudaFree(dataD);
	cudaFree(evensD);
	cudaFree(oddsD);
	cudaFree(evensCountD);
	cudaFree(oddsCountD);

	printf("Number of evens: %d, odds: %d\n", evensCount, oddsCount);

	printArray(evens, evensCount, "Evens");
	printArray(odds, oddsCount, "Odds");
	
	cudaFreeHost(data);
	cudaFreeHost(odds);
	cudaFreeHost(evens);

	return 0;
}
