
#include <stdio.h>
#include <pthread.h>

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

__global__ void kernelGetMax(int *out, int *data)
{
	atomicMax(out, data[threadIdx.x]);
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

struct threadData {
	cudaEvent_t event;
	cudaStream_t stream;
	int *countD;
	int *divisiblesD;
	int divisor;
};

void *callback(void *arg)
{
	struct threadData *threadData = (struct threadData*)arg;
	int *count = NULL;
	int *divisibles = NULL;

	cudaMallocHost(&divisibles, 256 * sizeof(int));
	cudaMallocHost(&count, sizeof(int));
	
	cudaEventSynchronize(threadData->event);

	printf("Divisor %d is done\n", threadData->divisor);

	cudaMemcpyAsync(count, threadData->countD, sizeof(int), cudaMemcpyDeviceToHost, threadData->stream);
	cudaMemcpyAsync(divisibles, threadData->divisiblesD, *threadData->countD * sizeof(int), cudaMemcpyDeviceToHost, threadData->stream);

	cudaStreamSynchronize(threadData->stream);
	cudaEventDestroy(threadData->event);
	cudaStreamDestroy(threadData->stream);
	printf("Done. There are %d elements divisible by %d\n", *count, threadData->divisor);

	cudaFreeHost(count);
	cudaFreeHost(divisibles);

	return NULL;
}

int main(int argc, char **argv)
{
	int *data = NULL;
	int *dataD = NULL;
	int **divisiblesD = NULL;
	int *countsD = NULL;
	cudaStream_t *streams = NULL;
	cudaEvent_t *events = NULL;
	pthread_t *thread_ids = NULL;

	if (argc < 4) {
		printf("Must supply length, max value, and a list of numbers\n");
		return 1;
	}

	int arrayLength = atoi(argv[1]);
	int maxValue = atoi(argv[2]);
	int arraySize = arrayLength * sizeof(*data);
	int numDivisors = argc - 3;

	// List of lists. Each lists hold numbers divisible by the corresponding divisor
	divisiblesD = (int**)malloc(numDivisors * sizeof(*divisiblesD));

	// List of events and streams, one for each divisor
	streams = (cudaStream_t*)malloc(numDivisors * sizeof(*streams));
	events = (cudaEvent_t*)malloc(numDivisors * sizeof(*events));

	thread_ids = (pthread_t*)malloc(numDivisors * sizeof(*thread_ids));

	// List of counts; length of each list in `lists`
	countsD = (int*)malloc(numDivisors * sizeof(*countsD));

	// For each divisor, allocate memory to store the numbers divisible by the divisor
	for (int i = 0; i < numDivisors; i++) {
		cudaMallocHost(divisiblesD + i, arraySize);
	}

	// Generate input data
	cudaMallocHost(&data, arraySize);
	cudaMalloc(&dataD, arraySize);
	generate_randoms(data, arrayLength, maxValue);
	printArray(data, arrayLength, "Data");
	cudaMemcpy(dataD, data, arraySize, cudaMemcpyHostToDevice);

	for (int i = 0; i < numDivisors; i++) {
		int currentDivisor = atoi(argv[3 + i]);

		cudaStreamCreate(streams + i);

		kernelGetDivisible<<<1, arrayLength, currentDivisor * sizeof(int), streams[i]>>>(divisiblesD[i], countsD + i, dataD, currentDivisor);

		cudaEventCreate(events + i);

		struct threadData *threadData = (struct threadData*)malloc(sizeof(struct threadData));
		threadData->event = events[i];
		threadData->stream = streams[i];
		threadData->countD = countsD + i;
		threadData->divisiblesD = divisiblesD[i];
		threadData->divisor = currentDivisor;

		pthread_create(thread_ids + i, NULL, callback, (void*)threadData);
	}

	for (int i = 0; i < numDivisors; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	// Free memory
	cudaFreeHost(data);
	cudaFree(dataD);

	for (int i = 0; i < numDivisors; i++) {
		cudaFree(divisiblesD[i]);
	}

	free(divisiblesD);
	free(countsD);
	free(streams);

	return 0;
}
