#ifndef FIELD_H
#define FIELD_H

static const unsigned FIELD_PIXEL_WIDTH_MAX = 1024;

// Just in case I decide to change the value type of the perlin noise
typedef float pixel_t;

/**
 * Creates a square grid of perlin noise called a "Field"
 * 
 * @description This function is basically a wrapper around the perlin noise generator function.
 *              Its purpose is to decouple the perlin noise generator itself from the reason it's
 *              being used. Namely, this function restricts the input to a square grid. Also
 *              abstracts the generator from being CUDA-implemented. I coined the term "field" to
 *              mean a map of perlin noise to be interpreted as a videogame map, like minecraft.
 *
 *              This function is deterministic in that identical inputs will yield identical outputs.
 *
 * @param fieldOut   output matrix of perlin noise
 * @param seed       global seed for the perlin noise
 * @param pixelWidth height/width of the perlin noise
 * @param x          x-coordinate to being the generation.
 * @param y          y-coordinate to being the generation.
 * @param octaves    number of octaves of perlin noise. TODO: Unimplemented
 * @param fieldOut   output field of perlin noise
 */
int createField(pixel_t *fieldOut, long seed, unsigned pixelWidth, long x, long y, unsigned octaves);

/**
 * Allocates space for a field.
 *
 * @param  pixelWidth height/width of the field. Must be a power of 2 and <= FIELD_PIXEL_WIDTH_MAX
 * @return pointer    to a buffer to use with createField
 */
pixel_t *fieldAlloc(unsigned pixelWidth);

/**
 * Frees memory allocated with fieldAlloc
 *
 * @param field pointer to memory allocated with fieldAlloc
 */
void fieldFree(pixel_t *field);

#endif
