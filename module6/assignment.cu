
#include <stdio.h>
#include <pthread.h>

/**
 * This kernel searches for all ints in `data` that are divisible by `divisor`, and enqueues them
 * into `divisiblesOut`. Also copies the number of them found into `countOut`
 */
__global__ void kernelGetDivisible(int *divisiblesOut, int *countOut, int *data, int divisor)
{
	// Length should equal divisor. The 0th elements is the count of divisible elements. The rest are "garbage"
	extern __shared__ int count[];

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

/**
 * Finds the largest int in `data` and copies it into `out`
 */
__global__ void kernelGetMax(int *out, int *data)
{
	atomicMax(out, data[threadIdx.x]);
}


__host__ void getRandomInts(int *out, int size, int maxValue)
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

/**
 * This runs in a thread. Takes a list of ints and gets the maximum element
 */
void *callback(void *arg)
{
	struct threadData *args = (struct threadData*)arg;
	int *count = NULL;
	int *divisibles = NULL;
	int *maxDivisible = NULL;
	int *maxDivisibleD = NULL;

	cudaMallocHost(&maxDivisible, sizeof(int));
	cudaMalloc(&maxDivisibleD, sizeof(int));

	cudaEventSynchronize(args->event);

	cudaMallocHost(&count, sizeof(int));
	cudaMemcpy(count, args->countD, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMallocHost(&divisibles, *count * sizeof(int));
	
	kernelGetMax<<<1, *args->countD, 0, args->stream>>>(maxDivisibleD, args->divisiblesD);
	cudaMemcpyAsync(divisibles, args->divisiblesD, *args->countD * sizeof(int), cudaMemcpyDeviceToHost, args->stream);
	cudaMemcpyAsync(maxDivisible, maxDivisibleD, sizeof(int), cudaMemcpyDeviceToHost, args->stream);

	cudaStreamSynchronize(args->stream);
	cudaEventDestroy(args->event);
	cudaStreamDestroy(args->stream);

	char msg[256];
	sprintf(msg, "There are %d elements divisible by %d (max %d)", *count, args->divisor, *maxDivisible);
	printArray(divisibles, *count, msg);

	cudaFreeHost(count);
	cudaFreeHost(maxDivisible);
	cudaFreeHost(divisibles);
	cudaFree(maxDivisibleD);

	return NULL;
}

/**
 * argv will have 3+ elements:
 *    number of elements, maximum value of element, and 1 or more divisors
 * Program will find all the elements divisible by each divisor, and then find the maximum.
 * This program is is probably overly complicated for what it does, but this is me having a hammer and finding a nail.
 */
int main(int argc, char **argv)
{
	int *data = NULL;
	int *dataD = NULL;

	if (argc < 4) {
		printf("Must supply length, max value, and a list of numbers\n");
		return 1;
	}

	int arrayLength = atoi(argv[1]);
	int maxValue = atoi(argv[2]);
	int arraySize = arrayLength * sizeof(*data);
	int numDivisors = argc - 3;

	// List of lists. Each lists hold numbers divisible by the corresponding divisor
	int **divisiblesD = (int**)malloc(numDivisors * sizeof(*divisiblesD));

	// List of events and streams, one for each divisor
	cudaStream_t *streams = (cudaStream_t*)malloc(numDivisors * sizeof(*streams));
	cudaEvent_t *events = (cudaEvent_t*)malloc(numDivisors * sizeof(*events));

	pthread_t *thread_ids = (pthread_t*)malloc(numDivisors * sizeof(*thread_ids));

	// List of counts; length of each list in `lists`
	int *countsD = (int*)malloc(numDivisors * sizeof(*countsD));

	// For each divisor, allocate memory to store the numbers divisible by the divisor
	for (int i = 0; i < numDivisors; i++) {
		cudaMallocHost(divisiblesD + i, arraySize);
	}

	// Generate input data
	cudaMallocHost(&data, arraySize);
	cudaMalloc(&dataD, arraySize);
	getRandomInts(data, arrayLength, maxValue);
	printArray(data, arrayLength, "Data");
	cudaMemcpy(dataD, data, arraySize, cudaMemcpyHostToDevice);

	// Loop through all divisisors, launch a kernel/stream for each and create a thread to manage the stream
	for (int i = 0; i < numDivisors; i++) {
		int currentDivisor = atoi(argv[3 + i]);

		cudaStreamCreate(streams + i);  // Clean this up in a thread

		// Get all ints divisible by `currentDivisor`. Do not wait for it to complete here
		kernelGetDivisible<<<1, arrayLength, currentDivisor * sizeof(int), streams[i]>>>(divisiblesD[i], countsD + i, dataD, currentDivisor);

		cudaEventCreate(events + i);  // Clean this up in a thread

		// Create a callback thread that will wait for this stream to complete, then perform more processing on the data
		// Maybe this is overly complicated.
		struct threadData *arg = (struct threadData*)malloc(sizeof(struct threadData));
		arg->event = events[i];  // Event for thread to wait on
		arg->stream = streams[i];  // Stream the thread will manage
		arg->countD = countsD + i;  // Where GPU will put the count of elements
		arg->divisiblesD = divisiblesD[i];  // Where GPU will put all the ints divisible by `currentDivisor`
		arg->divisor = currentDivisor;

		pthread_create(thread_ids + i, NULL, callback, (void*)arg);  // Launch thread
	}

	// Wait for threads to complete
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
	free(events);
	free(thread_ids);

	return 0;
}
