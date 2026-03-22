#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <cuda_gl_interop.h>

void convertNoiseToUchar3(uchar3 *pixels, const float *noise, unsigned pixelWidth);

#endif
