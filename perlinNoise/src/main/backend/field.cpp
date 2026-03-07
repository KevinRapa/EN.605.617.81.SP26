
#include <field.h>
#include <perlin.h>

#include <unordered_set>
#include <functional>

long createFieldSeed(long worldSeed, int x, int y)
{
	std::hash<long> hasher;
	return hasher(worldSeed) ^ hasher(static_cast<long>(x)) ^ hasher(static_cast<long>(y));
}

pixel_t *fieldAlloc()
{
	return new pixel_t[FIELD_DIM * FIELD_DIM * CHUNK_DIM * CHUNK_DIM];
}

int createField(pixel_t *fieldOut, long seed, int x, int y)
{
	if (NULL == fieldOut) {
		return FAILURE;
	}

	long fieldSeed = createFieldSeed(seed, x, y);

	return SUCCESS;
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
