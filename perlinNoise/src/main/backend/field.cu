
#include "field.h"
#include "perlin.h"

#include <cstdlib>
#include <iostream>
#include <limits>

static bool perlinInitialized = false;

static const unsigned FIELD_PIXEL_WIDTH_MAX = 1024;

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

void convertNoiseToUchar3(uchar3 *pixels, const float *noise, unsigned pixelWidth)
{
	float minFloat = std::numeric_limits<float>::max();
	float maxFloat = std::numeric_limits<float>::lowest();
	float maxAlpha = std::numeric_limits<unsigned char>::max();

	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		if (noise[i] < minFloat) {
			minFloat = noise[i];
		}
		if (noise[i] > maxFloat) {
			maxFloat = noise[i];
		}
	}

	maxFloat -= minFloat;

	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		float percentOfMax = (noise[i] - minFloat) / maxFloat;

		unsigned char alpha = static_cast<unsigned char>(maxAlpha * percentOfMax);
		pixels[i] = make_uchar3(alpha, alpha, alpha);
	}
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

	float *noise = new float[pixelWidth * pixelWidth];

	int ret = perlin(noise, seed, x, y, pixelWidth, pixelWidth, octaves);

	convertNoiseToUchar3(fieldOut, noise, pixelWidth);

#if 1
	for (int i = 0; i < pixelWidth * pixelWidth; i++) {
		if (i % pixelWidth == 0) printf("\n");
		printf("% 2.3f, ", noise[i]);
	}
#endif

	delete[] noise;

	return ret;
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
