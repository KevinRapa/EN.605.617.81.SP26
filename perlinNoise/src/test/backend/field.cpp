
#include <field.h>

#include <iostream>

bool fieldAllocTest()
{
	pixel_t *grid = fieldAlloc(256);

	bool pass = NULL != grid;
	
	printf("%s: %s\n", __func__, pass ? "PASS" : "FAIL");

	fieldFree(grid);

	return pass;
}

int main()
{
	fieldAllocTest();
}
