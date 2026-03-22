
#ifndef FIELD_H
#define FIELD_H

#include <cuda_gl_interop.h>

typedef uchar3 pixel_t;

void convertNoiseToUchar3(uchar3 *pixels, const float *noise, unsigned pixelWidth);

int createField(pixel_t *fieldOut, long seed, unsigned pixelWidth, long x, long y, unsigned octaves);

pixel_t *fieldAlloc(unsigned pixelWidth);

void fieldFree(pixel_t *field);

#endif
