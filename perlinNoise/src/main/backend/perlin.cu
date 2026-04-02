
#include "crc64.h"
#include "perlin.h"

#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Number of values wide a single chunk in a grid of perlin noise is. This is not a standard value.
static const unsigned CHUNK_DIM = 32;

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
	float vect = generateVector(seed, (blockIdx.x * blockDim.x) + xCoord + threadIdx.x,
	                                  (blockIdx.y * blockDim.y) + yCoord + threadIdx.y);

	vectorsOut[((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x)] = vect;
}

__device__ float computeDotProduct(float horizontalOffsetMagnitude, float verticalOffsetMagnitude, float gradientVectorAngle)
{
	return (horizontalOffsetMagnitude * cosf(gradientVectorAngle)) + (verticalOffsetMagnitude * sinf(gradientVectorAngle));
}

__device__ float linearInterpolate(float percent, float n1, float n2)
{
	return (1.0 - percent) * n1 + percent * n2;
}

__global__ void generatePerlinNoise(float *noiseOut, float *vectorMap, int vectorFieldWidth)
{
	int chunkIdxX = blockIdx.x;
	int chunkIdxY = blockIdx.y;
	int pixelIdxX = threadIdx.x;
	int pixelIdxY = threadIdx.y;
	int globalThreadIdx = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

	float thetaUL = vectorMap[chunkIdxY * vectorFieldWidth + chunkIdxX];
	float thetaUR = vectorMap[chunkIdxY * vectorFieldWidth + (chunkIdxX + 1)];
	float thetaLL = vectorMap[(chunkIdxY + 1) * vectorFieldWidth + chunkIdxX];
	float thetaLR = vectorMap[(chunkIdxY + 1) * vectorFieldWidth + (chunkIdxX + 1)];

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
	float percentUD = static_cast<float>(threadIdx.y) / static_cast<float>(blockDim.y);

	float interpBottom = linearInterpolate(percentLR, dotProductLL, dotProductLR);
	float interpTop = linearInterpolate(percentLR, dotProductUL, dotProductUR);

	noiseOut[globalThreadIdx] = linearInterpolate(percentUD, interpTop, interpBottom);
}

int perlinInit()
{
	return cudaMemcpyToSymbol(crctab64Device, crctab64, sizeof(crctab64));
}

void checkLastError()
{
	cudaError_t err = cudaGetLastError();

	if (err != 0) {
		printf("Error calling kernel: %s\n", cudaGetErrorString(err));
	}
}

int perlin(float *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves)
{
	if (yDim % CHUNK_DIM != 0 || xDim % CHUNK_DIM != 0) {
		fprintf(stderr, "%s: dimensions of noise grid must be a multiple of %u\n", __func__, CHUNK_DIM);
		return EXIT_FAILURE;
	}

	unsigned gridDimY = yDim / CHUNK_DIM;
	unsigned gridDimX = xDim / CHUNK_DIM;

	dim3 vectorFieldGridSize(2, 2);
	dim3 vectorFieldBlockSize(gridDimX, gridDimY);

	dim3 perlinGridDim(gridDimX, gridDimY);
	dim3 perlinChunkDim(CHUNK_DIM, CHUNK_DIM);

	// Plus 1 here because each float is a vector at a corner of a chunk, so there are gridDimX+1 and gridDimY+1 corners in each dimension
	thrust::device_vector<float> vectorMapD((gridDimX * 2) * (gridDimY * 2));
	thrust::device_vector<float> pixelsD(xDim * yDim);

	generateVectorField<<<vectorFieldGridSize, vectorFieldBlockSize>>>(thrust::raw_pointer_cast(vectorMapD.data()), seed, xCoord, yCoord);
	checkLastError();
	generatePerlinNoise<<<perlinGridDim, perlinChunkDim>>>(thrust::raw_pointer_cast(pixelsD.data()), thrust::raw_pointer_cast(vectorMapD.data()), gridDimX * 2);
	checkLastError();

	thrust::copy(pixelsD.begin(), pixelsD.end(), noiseOut);

	return EXIT_SUCCESS;
}
