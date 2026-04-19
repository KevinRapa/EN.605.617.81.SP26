
#include "crc64.h"
#include "perlin.h"

#include <cmath>
#include <cublas_v2.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

__constant__ DWORD64 crctab64Device[256];

static cublasHandle_t handle;

/**
 * Generates a random gradient vector. Returns angle in radians
 */
__device__ float generateVector(long worldSeed, long xCoord, long yCoord, unsigned octaveNum)
{
	long variables[] = { worldSeed, xCoord, yCoord, octaveNum };
	char *bytes = reinterpret_cast<char *>(variables);
	DWORD64 crc = CRC_INIT;

	for (int i = 0; i < sizeof(variables); i++) {
		// See https://stackoverflow.com/questions/45625145/why-does-perlin-noise-use-a-hash-function-rather-than-computing-random-values
		// For reasoning behind why I am using a hash function instead of curand to generate a random gradient vector
		crc = crctab64Device[(crc ^ bytes[i]) & 0xff] ^ (crc >> 8);
	}

	float radians = static_cast<float>(crc % 360) * (M_PI / 180.0);

	return radians;
}

/**
 * Generates a matrix of random gradient vectors
 */
__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord, unsigned octaveNum)
{
	float vect = generateVector(seed, (blockIdx.x * blockDim.x) + xCoord + threadIdx.x,
	                                  (blockIdx.y * blockDim.y) + yCoord + threadIdx.y, octaveNum);

	vectorsOut[((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x)] = vect;
}

/**
 * Computes the dot product of a gradient vector with a pixel's distance from the edges of a chunk
 *
 * gradientVectorAngle is the angle of the gradient vector, and the offsets are the horizontal/vertical distance
 * in pixels from the edges of the chunk. Those edges' intersection is where the gradient vector lies
 */
__device__ float computeDotProduct(float horizontalOffsetMagnitude, float verticalOffsetMagnitude, float gradientVectorAngle)
{
	return (horizontalOffsetMagnitude * cosf(gradientVectorAngle)) + (verticalOffsetMagnitude * sinf(gradientVectorAngle));
}

/**
 * Interpolation function used to combine dot products of each pixel. This gives satisfactory results,
 * but should be upgraded to a cubic function for more smoothness.
 */
__device__ float linearInterpolate(float percent, float n1, float n2)
{
	return (1.0 - percent) * n1 + percent * n2;
}

/**
 * Main kernel that generates perlin noise
 */
__global__ void generatePerlinNoise(float *noiseOut, float *vectorMap, int vectorFieldWidth)
{
	// This kernel is run for each pixel in the perlin noise map

	int chunkIdxX = blockIdx.x;
	int chunkIdxY = blockIdx.y;
	int pixelIdxX = threadIdx.x;
	int pixelIdxY = threadIdx.y;
	int globalThreadIdx = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

	// Find the 4 gradient vectors surrounding this pixel
	float thetaUL = vectorMap[chunkIdxY * vectorFieldWidth + chunkIdxX];
	float thetaUR = vectorMap[chunkIdxY * vectorFieldWidth + (chunkIdxX + 1)];
	float thetaLL = vectorMap[(chunkIdxY + 1) * vectorFieldWidth + chunkIdxX];
	float thetaLR = vectorMap[(chunkIdxY + 1) * vectorFieldWidth + (chunkIdxX + 1)];

	// Compute offsets of the pixel from the edges to use to compute the dot product
	float offsetFromLeftEdge = -pixelIdxX;
	float offsetFromTopEdge = pixelIdxY;
	float offsetFromRightEdge = static_cast<int>(blockDim.x) - pixelIdxX - 1.0;
	float offsetFromBottomEdge = -(static_cast<int>(blockDim.y) - pixelIdxY - 1.0);

	// Compute dot products
	float dotProductUL = computeDotProduct(offsetFromLeftEdge, offsetFromTopEdge, thetaUL);
	float dotProductUR = computeDotProduct(offsetFromRightEdge, offsetFromTopEdge, thetaUR);
	float dotProductLL = computeDotProduct(offsetFromLeftEdge, offsetFromBottomEdge, thetaLL);
	float dotProductLR = computeDotProduct(offsetFromRightEdge, offsetFromBottomEdge, thetaLR);

	// Linearly interpolate the values and store
	float percentLR = static_cast<float>(threadIdx.x) / static_cast<float>(blockDim.x);
	float percentUD = static_cast<float>(threadIdx.y) / static_cast<float>(blockDim.y);

	float interpBottom = linearInterpolate(percentLR, dotProductLL, dotProductLR);
	float interpTop = linearInterpolate(percentLR, dotProductUL, dotProductUR);

	noiseOut[globalThreadIdx] = linearInterpolate(percentUD, interpTop, interpBottom);
}

int perlinInit()
{
	cublasCreate(&handle);

	// Copy the crc 64 table to device memory for generating gradient vectors
	return cudaMemcpyToSymbol(crctab64Device, crctab64, sizeof(crctab64));
}

void perlinDeinit()
{
	cublasDestroy(handle);
}

void checkLastError()
{
	cudaError_t err = cudaGetLastError();

	if (err != 0) {
		printf("Error calling kernel: %s\n", cudaGetErrorString(err));
	}
}

int perlinDevice(thrust::device_vector<float> *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves, cudaStream_t stream)
{
	unsigned currentGridDim = 2;
	unsigned currentChunkDim = CHUNK_DIM;

	// Scales x/y coords. This is only needed to make sure the higher octaves move with the lower ones. This is because the block sizes
	// for higher octaves are smaller, so without this scale, the higher octaves will appear to move slower when panning the viewer. In other
	// words, the higher octaves will appear to be layers moving in the background.
	unsigned scale = 1;

	thrust::fill(thrust::cuda::par.on(stream), noiseOut->begin(), noiseOut->end(), 0);

	thrust::device_vector<float> pixelsD(xDim * yDim);

	for (unsigned o = 0; o < octaves && currentChunkDim > 1; o++) {
		// Number of chunks in each dimension. See comment below for what a chunk is
		unsigned gridDimY = yDim / currentChunkDim;
		unsigned gridDimX = xDim / currentChunkDim;

		dim3 vectorFieldGridSize(currentGridDim, currentGridDim);
		dim3 vectorFieldBlockSize(CHUNK_DIM, CHUNK_DIM);

		dim3 perlinGridDim(gridDimX, gridDimY);
		dim3 perlinChunkDim(currentChunkDim, currentChunkDim);

		// Generating a vector map here about twice the size it needs to be. Strictly speaking, a size
		// of (gridDimX+1) x (gridDimY+1) is sufficient, because given a grid X squares wide, there are X+1 corners
		// However, the math is easier if I just do double the size. There's probably a better way to do this, but this was easy enough
		thrust::device_vector<float> vectorMapD((gridDimX * 2) * (gridDimY * 2));

		// First generate a grid of vectors. The grid of pixels is divided into chunks, where each chunk is 32 pixels wide/tall. Each
		// Chunk is a square with a gradient vector at each corner.
		generateVectorField<<<vectorFieldGridSize, vectorFieldBlockSize, 0, stream>>>(
		    thrust::raw_pointer_cast(vectorMapD.data()),
		    seed, xCoord * scale, yCoord * scale, o
		);

		// Generate the perlin noise using the vector field
		generatePerlinNoise<<<perlinGridDim, perlinChunkDim, 0, stream>>>(
		    thrust::raw_pointer_cast(pixelsD.data()),
		    thrust::raw_pointer_cast(vectorMapD.data()),
		    gridDimX * 2
		);

		// With each octave, we generate noise double the granularity and add it to the final
		// result. This means double the number of gradient vectors are made each generation.
		const float SCALAR = 1.0;
		cublasSetStream(handle, stream);
		cublasSaxpy(handle, xDim * yDim, &SCALAR, thrust::raw_pointer_cast(pixelsD.data()), 1, thrust::raw_pointer_cast(noiseOut->data()), 1);

		scale *= 2;
		currentGridDim *= 2;
		currentChunkDim /= 2;
	}

	return EXIT_SUCCESS;
}

int perlin(float *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves)
{
	if (yDim % CHUNK_DIM != 0 || xDim % CHUNK_DIM != 0) {
		fprintf(stderr, "%s: dimensions of noise grid must be a multiple of %u\n", __func__, CHUNK_DIM);
		return EXIT_FAILURE;
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	thrust::device_vector<float> pixelsFinalD(xDim * yDim);

	perlinDevice(&pixelsFinalD, seed, xCoord, yCoord, xDim, yDim, octaves, stream);

	cudaStreamDestroy(stream);

	thrust::copy(pixelsFinalD.begin(), pixelsFinalD.end(), noiseOut);

	return EXIT_SUCCESS;
}
