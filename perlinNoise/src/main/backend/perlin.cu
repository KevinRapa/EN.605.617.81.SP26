
#include "crc64.h"
#include "perlin.h"

#include <cmath>

__constant__ DWORD64 crctab64Device[256];

__device__ float generateVector(long worldSeed, long xCoord, long yCoord)
{
	long variables[] = { worldSeed, xCoord, yCoord };
	char *bytes = reinterpret_cast<char *>(variables);
	DWORD64 crc = CRC_INIT;

	for (int i = 0; i < sizeof(variables); i++) {
		crc = crctab64Device[(crc ^ bytes[i]) & 0xff] ^ (crc >> 8);
	}

	float radians = static_cast<float>(crc % 360) * (M_PI / 180.0);

	return radians;
}

__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord)
{
	vectorsOut[(threadIdx.y * blockDim.x) + threadIdx.x] = generateVector(seed, xCoord + threadIdx.x, yCoord + threadIdx.y);
}

__global__ void generatePerlinNoise(float *noiseOut, float *vectorMap)
{
}

int perlinInit()
{
	cudaError_t ret = cudaMemcpyToSymbol(crctab64Device, crctab64, sizeof(crctab64));

	if (ret != cudaSuccess) {
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int perlin(float *pixelsOut, long seed, long xCoord, long yCoord, int numChunksXY, int numPixelsXY, unsigned octaves)
{
	float *vectorMapD;
	float *pixelsD;

	int numPixels = numChunksXY * numChunksXY * numPixelsXY * numPixelsXY;

	dim3 vectorFieldGridSize(1, 1);
	dim3 vectorFieldBlockSize(numChunksXY + 1, numChunksXY + 1);

	dim3 perlinGridDim(numChunksXY, numChunksXY);
	dim3 perlinChunkDim(numPixelsXY, numPixelsXY);

	// Plus 1 here because each float is a vector at a corner of a chunk, so there are numChunksXY+1 corners in each dimension
	cudaMalloc(&vectorMapD, (numChunksXY + 1) * (numChunksXY + 1) * sizeof(*vectorMapD));
	cudaMalloc(&pixelsD, numPixels * sizeof(*pixelsD));

	generateVectorField<<<vectorFieldGridSize, vectorFieldBlockSize>>>(vectorMapD, seed, xCoord, yCoord);
	generatePerlinNoise<<<perlinGridDim, perlinChunkDim>>>(pixelsD, vectorMapD);

	cudaMemcpy(pixelsOut, pixelsD, numPixels * sizeof(*pixelsD), cudaMemcpyDeviceToHost);

	cudaFree(vectorMapD);
	cudaFree(pixelsD);

	return EXIT_SUCCESS;
}
