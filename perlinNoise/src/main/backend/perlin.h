#ifndef PERLIN_H
#define PERLIN_H

static const unsigned FIELD_DIM_MAX = 8;
static const unsigned CHUNK_DIM_MAX= 32;

int perlinInit();

int perlin(float *pixelsOut, long seed, long xCoord, long yCoord, int numChunksXY, int numPixelsXY, unsigned octaves);

#endif
