
#include <field.h>
#include <perlin.h>
#include <crc64.h>

#include <unordered_set>

long createFieldSeed(long worldSeed, int x, int y)
{
	long bytes[] = { worldSeed, static_cast<long>(x), static_cast<long>(y) };
	DWORD64 ret = 0;
	GetCRC64(ret, reinterpret_cast<const unsigned char *>(bytes), sizeof(bytes));
	return ret;
}

pixel_t *fieldAlloc()
{
	return new pixel_t[FIELD_DIM * FIELD_DIM * CHUNK_DIM * CHUNK_DIM];
}

int createField(pixel_t *fieldOut, long seed, int x, int y)
{
	if (nullptr == fieldOut) {
		return FAILURE;
	}

	long fieldSeed = createFieldSeed(seed, x, y);

	return SUCCESS;
}

void fieldFree(pixel_t *field)
{
	delete[] field;
}
