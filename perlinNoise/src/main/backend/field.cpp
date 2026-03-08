
#include "field.h"
#include "perlin.h"

#include <cstdlib>

static bool perlinInitialized = false;

pixel_t *fieldAlloc()
{
	return new pixel_t[FIELD_DIM_MAX * FIELD_DIM_MAX * CHUNK_DIM_MAX * CHUNK_DIM_MAX];
}

int createField(pixel_t *fieldOut, long seed, long x, long y, unsigned octaves)
{
	if (nullptr == fieldOut) {
		return EXIT_FAILURE;
	}

	if (!perlinInitialized) {
		if (!perlinInit()) {
			return EXIT_FAILURE;
		}
	}

	return perlin(fieldOut, seed, x, y, FIELD_DIM_MAX, CHUNK_DIM_MAX, octaves);
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
