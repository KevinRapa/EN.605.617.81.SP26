
#include <iostream>

#include "field.h"

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Require a numerical seed\n");
		return EXIT_FAILURE;
	}

	long seed = atol(argv[1]);	
	unsigned pixelDim = 256;

	if (argc > 2) {
		pixelDim = atoi(argv[2]);
	}

	pixel_t *field = fieldAlloc(pixelDim);

	if (field == nullptr) {
		fprintf(stderr, "Failed to allocate memory for field\n");
		return EXIT_FAILURE;
	}

	int success = createField(field, seed, pixelDim, 0, 0, 4);

	fieldFree(field);

	return success;
}
