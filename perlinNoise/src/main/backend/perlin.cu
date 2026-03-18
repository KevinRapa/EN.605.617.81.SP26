
#include "crc64.h"
#include "perlin.h"

#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__constant__ DWORD64 crctab64Device[256];

static bool isPowerOfTwo(unsigned x)
{
	return x && !(x & (x-1));
}

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

__device__ float computeDotProduct(float horizontalOffsetMagnitude, float verticalOffsetMagnitude, float gradientVectorAngle)
{
	return (horizontalOffsetMagnitude * cosf(gradientVectorAngle)) + (verticalOffsetMagnitude * sinf(gradientVectorAngle));
}

__device__ float linearInterpolate(float percent, float n1, float n2)
{
	return (1.0 - percent) * n1 + percent * n2;
}

__global__ void generatePerlinNoise(float *noiseOut, float *vectorMap)
{
	int chunkIdxX = blockIdx.x;
	int chunkIdxY = blockIdx.y;
	int pixelIdxX = threadIdx.x;
	int pixelIdxY = threadIdx.y;
	int numChunksX = gridDim.x + 1;
	int globalThreadIdx = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

	float thetaUL = vectorMap[chunkIdxY * numChunksX + chunkIdxX];
	float thetaUR = vectorMap[chunkIdxY * numChunksX + (chunkIdxX + 1)];
	float thetaLL = vectorMap[(chunkIdxY + 1) * numChunksX + chunkIdxX];
	float thetaLR = vectorMap[(chunkIdxY + 1) * numChunksX + (chunkIdxX + 1)];

	float offsetFromLeftEdge = -pixelIdxX;
	float offsetFromTopEdge = pixelIdxY;
	float offsetFromRightEdge = static_cast<int>(blockDim.x) - pixelIdxX - 1.0;
	float offsetFromBottomEdge = -(static_cast<int>(blockDim.y) - pixelIdxY - 1.0);

#if 0
	printf("BLOCK(%d,%d) PIXEL(%d,%d)  UL:%f  UR:%f  LL:%f  LR:%f\n"
	       "offFromLeft:%f   offFromRight:%f  offFromTop:%f  offFromBottom:%f\n", 
               blockIdx.x, blockIdx.y, pixelIdxX, pixelIdxY, thetaUL, thetaUR, thetaLL, thetaLR,
	       offsetFromLeftEdge, offsetFromRightEdge, offsetFromTopEdge, offsetFromBottomEdge);
#endif

	float dotProductUL = computeDotProduct(offsetFromLeftEdge, offsetFromTopEdge, thetaUL);
	float dotProductUR = computeDotProduct(offsetFromRightEdge, offsetFromTopEdge, thetaUR);
	float dotProductLL = computeDotProduct(offsetFromLeftEdge, offsetFromBottomEdge, thetaLL);
	float dotProductLR = computeDotProduct(offsetFromRightEdge, offsetFromBottomEdge, thetaLR);

	float percentLR = static_cast<float>(threadIdx.x) / static_cast<float>(blockDim.x);
	float percentUD = (static_cast<float>(blockDim.y) - static_cast<float>(threadIdx.y)) / static_cast<float>(blockDim.y);

	float interpBottom = linearInterpolate(percentLR, dotProductLL, dotProductLR);
	float interpTop = linearInterpolate(percentLR, dotProductUL, dotProductUR);

	noiseOut[globalThreadIdx] = linearInterpolate(percentUD, interpTop, interpBottom);
}

int perlinInit()
{
	return cudaMemcpyToSymbol(crctab64Device, crctab64, sizeof(crctab64));
}

int perlin(float *pixelsOut, long seed, long xCoord, long yCoord, int numChunksXY, int numPixelsXY, unsigned octaves)
{
	if (!isPowerOfTwo(numChunksXY)) {
		fprintf(stderr, "%s: grid dimension must be a power of 2\n", __func__);
		return EXIT_FAILURE;
	}
	if (!isPowerOfTwo(numPixelsXY)) {
		fprintf(stderr, "%s: chunk dimension must be a power of 2\n", __func__);
		return EXIT_FAILURE;
	}

	int numPixels = numChunksXY * numChunksXY * numPixelsXY * numPixelsXY;

	dim3 vectorFieldGridSize(1, 1);
	dim3 vectorFieldBlockSize(numChunksXY + 1, numChunksXY + 1);

	dim3 perlinGridDim(numChunksXY, numChunksXY);
	dim3 perlinChunkDim(numPixelsXY, numPixelsXY);

	// Plus 1 here because each float is a vector at a corner of a chunk, so there are numChunksXY+1 corners in each dimension
	thrust::device_vector<float> vectorMapD((numChunksXY + 1) * (numChunksXY + 1));
	thrust::device_vector<float> pixelsD(numPixels);

	generateVectorField<<<vectorFieldGridSize, vectorFieldBlockSize>>>(thrust::raw_pointer_cast(vectorMapD.data()), seed, xCoord, yCoord);
	generatePerlinNoise<<<perlinGridDim, perlinChunkDim>>>(thrust::raw_pointer_cast(pixelsD.data()), thrust::raw_pointer_cast(vectorMapD.data()));

	thrust::copy(pixelsD.begin(), pixelsD.end(), pixelsOut);

	return EXIT_SUCCESS;
}
