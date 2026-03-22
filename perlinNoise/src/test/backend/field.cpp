
#include <field.h>

#include <cassert>
#include <iostream>

void fieldAllocTest()
{
	pixel_t *grid = fieldAlloc(256);

	assert(grid != nullptr);

	fieldFree(grid);
}

int main()
{
	fieldAllocTest();
}
