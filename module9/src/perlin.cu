
#include "perlin.h"

#include <cmath>
#include <iostream>

#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

static bool isPowerOfTwo(unsigned x)
{
	return x && !(x & (x-1));
}

// Generates a random 2D vector. Returns direction of vector from (0, 2*pi]; zero is excluded in curand_uniform
__device__ float generateVector(curandStateXORWOW_t *state)
{
	// Use curand_uniform so that any vector heading is equally likely. Interpret the value as a percent of 2*pi
	float percentOf2Pi = curand_uniform(state);

	float radians = (2.0 * M_PI) * percentOf2Pi;

	return radians;
}

// Creates a grid of random 2D vectors. Output must be the same given the same seed, xCoord, and yCoord
// vectorsOut - output grid
// seed   -     global seed for the generator
// xCoord -     X-coordinate of top-left corner of the grid we are generating
// yCoord -     Y-coordinate of top-left corner of the grid we are generating
__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord)
{
	curandStateXORWOW_t state;

	long globalThreadX = threadIdx.x + xCoord;
	long globalThreadY = threadIdx.y + yCoord;
	long seedModifier = (globalThreadX << 32) | globalThreadY;

	curand_init(seed ^ seedModifier, (threadIdx.y * blockDim.x) + threadIdx.x, 0, &state);

	vectorsOut[(threadIdx.y * blockDim.x) + threadIdx.x] = generateVector(&state);
}

// Helper function for computing the dot-product of a gradient vector and an offset vector
__device__ float computeDotProduct(float horizontalOffsetMagnitude, float verticalOffsetMagnitude, float gradientVectorAngle)
{
	return (horizontalOffsetMagnitude * cosf(gradientVectorAngle)) + (verticalOffsetMagnitude * sinf(gradientVectorAngle));
}

// Simple linear interpolation function for the final step in generating perlin noise
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

// Nothing to do for initializing the generator for now
int perlinInit()
{
	return 0;
}

// perlin noise generator function
// Calculates a grid of perlin noise. The grid is anchored at the top-left corner
// pixelsOut   - output grid. The size is numChunksXY^2 * numPixelsXY^2
// seed        - global seed for the perlin noise generator
// xCoord      - X-coordinate of the top-left corner of the grid
// yCoord      - Y-coordinate of the top-right corner of the grid
// numChunksXY - Number of chunks
// numPixelsXY - Number of "pixels" inside of each chunk
// octaves     - number of octaves for the perlin noise. Unimplemented feature.
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
