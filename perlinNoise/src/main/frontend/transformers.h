#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>

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
void convertNoiseToUchar3(uchar3 *pixels, const char *noise, unsigned pixelWidth);

/**
 * Converts three layers of perlin noise into a canvas of pixel colors
 *
 * @description Uses elevation and humidity values to determine biomes for each pixel.
 *              Currently supports snow cap, mountain, forest, plains, and river.
 *              Adds additional details such as trees and bushes, and colors mountains
 *              and rivers varying colors depending on elevation/depth
 *
 * @param pixels     output array.
 * @param elevation  perlin noise map representing elevation.
 * @param humidity   perlin noise map representing humidity.
 * @param details    perlin noise map for details. The values in this map are interpreted
 *                   as probabilities that a feature exists in the pixel, such as a tree.
 * @param pixelWidth x/y dimension of noise maps.
 */
void combineElevationAndHumdityLayers(thrust::device_vector<uchar3> *pixels,
                                      const thrust::device_vector<float> *elevation,
                                      const thrust::device_vector<float> *humidity,
                                      const thrust::device_vector<float> *details,
                                      unsigned pixelWidth);

#endif
