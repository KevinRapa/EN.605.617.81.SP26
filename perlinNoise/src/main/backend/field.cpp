
#include "field.h"
#include "perlin.h"

#include <cstdlib>
#include <iostream>

static bool perlinInitialized = false;

static const unsigned FIELD_PIXEL_WIDTH_MAX = 1024;

static const unsigned FIELD_DIM = 8;
static const unsigned CHUNK_DIM = 32;

static bool isPowerOfTwo(unsigned x)
{
	return x && !(x & (x-1));
}

static bool validateSize(unsigned pixelWidth)
{
	if (!isPowerOfTwo(pixelWidth)) {
		fprintf(stderr, "Field dimension must be a power of 2\n");
		return false;
	}

	if (pixelWidth > FIELD_PIXEL_WIDTH_MAX) {
		fprintf(stderr, "Field dimension is too large. Max is %u\n", FIELD_PIXEL_WIDTH_MAX);
		return false;
	}

	return true;
}

pixel_t *fieldAlloc(unsigned pixelWidth)
{
	if (!validateSize(pixelWidth)) {
		return nullptr;
	}

	return new pixel_t[pixelWidth * pixelWidth];
}

int createField(pixel_t *fieldOut, long seed, unsigned pixelWidth, long x, long y, unsigned octaves)
{
	if (!validateSize(pixelWidth)) {
		return EXIT_FAILURE;
	}

	if (!perlinInitialized) {
		int code = perlinInit();

		if (code != 0) {
			fprintf(stderr, "%s: failed to initialize Perlin Generator (%d)\n", __func__, code);
			return EXIT_FAILURE;
		}
	}

	int ret = perlin(fieldOut, seed, x, y, pixelWidth, pixelWidth, octaves);

#if 1
	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		if (i % pixelWidth == 0) printf("\n");
		printf("% 2.3f, ", fieldOut[i]);
	}
#endif

	return ret;
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
