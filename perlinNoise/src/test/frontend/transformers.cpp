
#include <transformers.h>

#include <cassert>

#include <cuda_gl_interop.h>

void testConvertNoiseToUchar3()
{
	uchar3 *pixels = new uchar3[4];
	unsigned char *p = reinterpret_cast<unsigned char *>(pixels);
	float noise[] = { 50.0, 60.0, 70.0, 80.0 };

	convertNoiseToUchar3(pixels, noise, 2);

	assert(p[0 * 3] == 0);
	assert(p[1 * 3] == 85);
	assert(p[2 * 3] == 170);
	assert(p[3 * 3] == 255);

	delete[] pixels;
}

void testConvertNoiseToUchar3_2()
{
	uchar3 *pixels = new uchar3[4];
	unsigned char *p = reinterpret_cast<unsigned char *>(pixels);
	float noise[] = { -10.0, 0.0, 10.0, 20.0 };

	convertNoiseToUchar3(pixels, noise, 2);

	assert(p[0 * 3] == 0);
	assert(p[1 * 3] == 85);
	assert(p[2 * 3] == 170);
	assert(p[3 * 3] == 255);

	delete[] pixels;
}

int main()
{
	testConvertNoiseToUchar3();
	testConvertNoiseToUchar3_2();
}
