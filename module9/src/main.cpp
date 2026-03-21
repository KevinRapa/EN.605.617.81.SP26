
#include <iostream>

#include "perlin.h"

static const unsigned FIELD_DIM = FIELD_DIM_MAX;
static const unsigned CHUNK_DIM = CHUNK_DIM_MAX;

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Require a numerical seed\n");
		return EXIT_FAILURE;
	}

	long seed = atol(argv[1]);	

	float *data = new float[FIELD_DIM * FIELD_DIM * CHUNK_DIM * CHUNK_DIM];

	perlinInit();

	int success = perlin(data, seed, 0, 0, FIELD_DIM, CHUNK_DIM, 0);

	for (int i = 0; i < FIELD_DIM * FIELD_DIM * CHUNK_DIM * CHUNK_DIM; i++) {
		if (i % (FIELD_DIM * CHUNK_DIM) == 0) printf("\n");
		printf("% 2.3f, ", data[i]);
	}

	delete[] data;

	return success;
}
