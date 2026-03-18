
#include "field.h"
#include "perlin.h"

#include <cstdlib>
#include <iostream>

static bool perlinInitialized = false;

static const unsigned FIELD_DIM = FIELD_DIM_MAX;
static const unsigned CHUNK_DIM = CHUNK_DIM_MAX;

pixel_t *fieldAlloc()
{
	return new pixel_t[FIELD_DIM * FIELD_DIM * CHUNK_DIM * CHUNK_DIM];
}

int createField(pixel_t *fieldOut, long seed, long x, long y, unsigned octaves)
{
	if (nullptr == fieldOut) {
		fprintf(stderr, "%s: input field memory is null\n", __func__);
		return EXIT_FAILURE;
	}

	if (!perlinInitialized) {
		int code = perlinInit();

		if (code != 0) {
			fprintf(stderr, "%s: failed to initialize Perlin Generator (%d)\n", __func__, code);
			return EXIT_FAILURE;
		}
	}

	int ret = perlin(fieldOut, seed, x, y, FIELD_DIM, CHUNK_DIM, octaves);

#if 1
	for (int i = 0; i < FIELD_DIM *FIELD_DIM * CHUNK_DIM * CHUNK_DIM; i++) {
		if (i % (FIELD_DIM * CHUNK_DIM) == 0) printf("\n");
		printf("% 2.3f, ", fieldOut[i]);
	}
#endif

	return ret;
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
