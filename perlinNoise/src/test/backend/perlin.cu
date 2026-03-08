
#include <perlin.h>
#include <cassert>

__device__ float generateVector(long worldSeed, long xCoord, long yCoord);
__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord);

__global__ void generateVectorCall(float* out, long seed, long x, long y)
{
	*out = generateVector(seed, x, y);
}

void generateVectorTest()
{
	long seed = 0x0123456789ABCDEF;
	float *vec1D, *vec2D;
	float vec1, vec2;

	cudaMalloc(&vec1D, sizeof(*vec1D));
	cudaMalloc(&vec2D, sizeof(*vec2D));
	
	generateVectorCall<<<1, 1>>>(vec1D, seed, 0, 0);
	generateVectorCall<<<1, 1>>>(vec2D, seed, 0, 0);

	cudaMemcpy(&vec1, vec1D, sizeof(*vec1D), cudaMemcpyDeviceToHost);
	cudaMemcpy(&vec2, vec2D, sizeof(*vec2D), cudaMemcpyDeviceToHost);

	assert(vec1 == vec2);

	generateVectorCall<<<1, 1>>>(vec1D, seed, 3, 4);
	generateVectorCall<<<1, 1>>>(vec2D, seed, 4, 3);

	cudaMemcpy(&vec1, vec1D, sizeof(*vec1D), cudaMemcpyDeviceToHost);
	cudaMemcpy(&vec2, vec2D, sizeof(*vec2D), cudaMemcpyDeviceToHost);

	assert(vec1 != vec2);

	cudaFree(vec1D);
	cudaFree(vec2D);
}

void generateVectorFieldTest()
{
	float *mapCenterD, *mapRightD, *mapDownD, *mapUpD;
	float *mapCenter, *mapRight, *mapDown, *mapUp;

	long seed = 0x0123456789ABCDEF;
	int numChunksXY = 8;
	int numCornersXY = numChunksXY + 1;
	int fieldSize = numCornersXY * numCornersXY * sizeof(*mapCenterD);
	dim3 blockSize(numCornersXY, numCornersXY);

	mapCenter = (float *)malloc(fieldSize);
	mapRight = (float *)malloc(fieldSize);
	mapDown = (float *)malloc(fieldSize);
	mapUp = (float *)malloc(fieldSize);
	cudaMalloc(&mapCenterD, fieldSize);
	cudaMalloc(&mapRightD, fieldSize);
	cudaMalloc(&mapDownD, fieldSize);
	cudaMalloc(&mapUpD, fieldSize);

	generateVectorField<<<dim3(1, 1), blockSize>>>(mapCenterD, seed, 0, 0);
	generateVectorField<<<dim3(1, 1), blockSize>>>(mapRightD, seed, numChunksXY, 0);
	generateVectorField<<<dim3(1, 1), blockSize>>>(mapDownD, seed, 0, numChunksXY);
	generateVectorField<<<dim3(1, 1), blockSize>>>(mapUpD, seed, 0, -numChunksXY);

	cudaMemcpy(mapCenter, mapCenterD, fieldSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mapRight, mapRightD, fieldSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mapDown, mapDownD, fieldSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(mapUp, mapUpD, fieldSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numCornersXY; i++) {
		assert(mapCenter[i * numCornersXY + (numCornersXY - 1)] == mapRight[i * numCornersXY]);
		assert(mapCenter[i] == mapUp[numCornersXY * (numCornersXY - 1) + i]);
		assert(mapCenter[numCornersXY * (numCornersXY - 1) + i] = mapDown[i]);
	}

	cudaFree(mapCenterD);
	cudaFree(mapRightD);
	cudaFree(mapDownD);
	cudaFree(mapUpD);
	free(mapCenter);
	free(mapRight);
	free(mapUp);
	free(mapDown);
}

int main()
{
	perlinInit();

	generateVectorTest();
	generateVectorFieldTest();
}
