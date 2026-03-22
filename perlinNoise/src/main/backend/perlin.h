#ifndef PERLIN_H
#define PERLIN_H

int perlinInit();

int perlin(float *noiseOut, long seed, long xCoord, long yCoord, unsigned xDim, unsigned yDim, unsigned octaves);

#endif
