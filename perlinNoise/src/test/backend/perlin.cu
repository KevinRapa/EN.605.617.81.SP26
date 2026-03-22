
#include <perlin.h>
#include <cassert>
#include <iostream>

__device__ float generateVector(long worldSeed, long xCoord, long yCoord);
__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord);
__global__ void generatePerlinNoise(float *noiseOut, float *vectorMap, int vectorFieldWidth);

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

void generatePerlinNoiseTest()
{
	float vectorField[8][8] = {
		{ 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07 },
		{ 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17 },
		{ 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27 },
		{ 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37 },
		{ 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47 },
		{ 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57 },
		{ 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67 },
		{ 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77 },
	};
	float *vectorFieldD = nullptr;

	float noiseOut[8][8] = { 0 };
	float *noiseOutD = nullptr;

	cudaMalloc(&vectorFieldD, sizeof(vectorField));
	cudaMalloc(&noiseOutD, sizeof(noiseOut));

	cudaMemcpy(vectorFieldD, vectorField, sizeof(vectorField), cudaMemcpyHostToDevice);

	generatePerlinNoise<<<dim3(4, 4), dim3(2, 2)>>>(noiseOutD, vectorFieldD, 8);

	cudaMemcpy(noiseOut, noiseOutD, sizeof(noiseOut), cudaMemcpyDeviceToHost);

	for ( int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			printf("% 2.3f, ", noiseOut[i][j]);
		}
		printf("\n");
	}

	cudaFree(vectorFieldD);
	cudaFree(noiseOutD);
}

int main()
{
	perlinInit();

	generateVectorTest();
	generateVectorFieldTest();
	generatePerlinNoiseTest();
}
