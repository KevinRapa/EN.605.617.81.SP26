#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <cuda_gl_interop.h>

/**
 * Converts a matrix of perlin noise, represented as floats, to a matrix of uchar3 representing pixel colors
 *
 * @description The output pixel values are grayscale. Their values are linearly correlated to the input
 *              floats using the maximum and minimum values. For example, if the minimum value in `noise`
 *              is 0.0 and the maximum value is 10.0, then a value of 5.0 will map to (255 / 2), 0.0 will
 *              map to 0, and 10.0 will map to 255.
 *
 * @param  pixels     output array.
 * @param  noise      input array. Must be aquare.
 * @param  pixelWidth x/y dimension of noise.
 * @return            A `pixelWidth` x `pixelWidth` matrix of uchar3
 */
void convertNoiseToUchar3(uchar3 *pixels, const float *noise, unsigned pixelWidth);

#endif
