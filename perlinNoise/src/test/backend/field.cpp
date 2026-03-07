
#include <field.h>

#include <iostream>

bool createFieldSeedTest(long worldSeed, int x, int y)
{
	long seed1 = createFieldSeed(worldSeed, x, y);
	long seed2 = createFieldSeed(worldSeed, x, y);
	bool pass = seed1 == seed2;

	printf("%s (%ld, %d, %d): %s (%ld, %ld)\n",
	       __func__, worldSeed, x, y, pass ? "PASS" : "FAIL", seed1, seed2);

	return pass;
}

bool fieldAllocTest()
{
	pixel_t *grid = fieldAlloc();

	bool pass = NULL != grid;
	
	printf("%s: %s\n", __func__, pass ? "PASS" : "FAIL");

	fieldFree(grid);

	return pass;
}

void createFieldTest()
{
}

int main()
{
	createFieldSeedTest(0x1234567890ABCDEF, 26, 37);
	createFieldSeedTest(0, 0, 0);
	createFieldSeedTest(-56747394, -26, -37);

	fieldAllocTest();
}
