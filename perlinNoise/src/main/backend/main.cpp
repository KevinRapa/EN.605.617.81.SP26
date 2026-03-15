
#include <iostream>

#include "field.h"

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Require a numerical seed\n");
		return EXIT_FAILURE;
	}

	long seed = atol(argv[1]);	

	pixel_t *field = fieldAlloc();

	int success = createField(field, seed, 0, 0, 4);

	fieldFree(field);

	return success;
}
