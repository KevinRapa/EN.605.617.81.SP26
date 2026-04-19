#ifndef PERLIN_H
#define PERLIN_H

#include <thrust/device_vector.h>

// Number of values wide a single chunk in a grid of perlin noise is. This is not a standard value.
static const unsigned CHUNK_DIM = 32;

/**
 * Initializes the perlin noise generator. Must be called before perlin
 */
int perlinInit();

/**
 * Cleans up resources allocated in generator
 */
void perlinDeinit();

/**
 * Generated a matrix of perlin noise. Identical inputs result in identical outputs
 *
 * @param  noiseOut output matrix of floats. The range of these values is undefined.
 * @param  seed     random seed
 * @param  xCoord   x-coordinate to begin the generation
 * @param  yCoord   y-coordinate to begin the generation
 * @param  xDim     horizontal dimension of the perlin noise. Must be multiple of CHUNK_DIM
 * @param  yDim     vertical dimension of the perlin noise. Must be multiple of CHUNK_DIM
 * @param  octaves  number of octaves to generate. TODO: Unimplemented
 * @return          EXIT_SUCCESS on success, EXIT_FAILURE on failure.
 */
int perlin(float *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves);

int perlinDevice(thrust::device_vector<float> *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves, cudaStream_t stream);

#endif
