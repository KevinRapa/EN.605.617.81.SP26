
#include <perlin.h>
#include <cassert>

__device__ float generateVector(long worldSeed, long xCoord, long yCoord);
__global__ void generateVectorField(float *vectorsOut, long seed, long xCoord, long yCoord);

__global__ void generateVectorCall(float* out, long seed, long x, long y)
{
	*out = generateVector(seed, x, y);
}

void generateVectorTest()
{
	long seed = 0x0123456789ABCDEF;
	float *vec1D, *vec2D;
	float vec1, vec2;

	cudaMalloc(&vec1D, sizeof(*vec1D));
	cudaMalloc(&vec2D, sizeof(*vec2D));
	
	generateVectorCall<<<1, 1>>>(vec1D, seed, 0, 0);
	generateVectorCall<<<1, 1>>>(vec2D, seed, 0, 0);

	cudaMemcpy(&vec1, vec1D, sizeof(*vec1D), cudaMemcpyDeviceToHost);
	cudaMemcpy(&vec2, vec2D, sizeof(*vec2D), cudaMemcpyDeviceToHost);

	assert(vec1 == vec2);

	generateVectorCall<<<1, 1>>>(vec1D, seed, 3, 4);
	generateVectorCall<<<1, 1>>>(vec2D, seed, 4, 3);

	cudaMemcpy(&vec1, vec1D, sizeof(*vec1D), cudaMemcpyDeviceToHost);
	cudaMemcpy(&vec2, vec2D, sizeof(*vec2D), cudaMemcpyDeviceToHost);

	assert(vec1 != vec2);

	cudaFree(vec1D);
	cudaFree(vec2D);
}

void generateVectorFieldTest()
{
}

int main()
{
	perlinInit();

	generateVectorTest();
}
