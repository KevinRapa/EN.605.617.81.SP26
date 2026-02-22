
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

#define ROWS_MAX 32
#define COLS_MAX 32

__constant__ int ADDEND[ROWS_MAX * COLS_MAX];

__global__ void kernel_matrix_madd(int *out, const int *a, int rowsA, int colsA, const int *b, int rowsB, int colsB)
{
	extern __shared__ int matrixCache[];

	// copy the first matrix into the first half of matrixCache,
	// then transpose the second matrix into the second half of matrixCache,
	// Unfortunate, but can't parallize here because
	// matrices may be larger than the number of threads
	// in this function (which is the size of the result)
	if (threadIdx.y == 0 && threadIdx.x == 0) {
		for (int i = 0; i < rowsA * colsA; i++) {
			matrixCache[i] = a[i];
		}
		for (int i = 0; i < rowsB; i++) {
			for (int j = 0; j < colsB; j++) {
				matrixCache[(rowsA * colsA) + (j * rowsB + i)] = b[i * colsB + j];
			}
		}
	}

	__syncthreads();

	int result = 0;

	// TODO: Bug here?
	for (int i = 0; i < colsA; i++) {
		for (int j = 0; j < rowsB; j++) {
			result += matrixCache[(threadIdx.y * colsA) + i] * matrixCache[(rowsA * colsA) + (threadIdx.x * rowsB) + j];
		}
	}

	out[threadIdx.y * blockDim.x + threadIdx.x] = result + ADDEND[threadIdx.y * blockDim.x + threadIdx.x];
}

void accelerated_matrix_madd(int *result, const int *A, int rowsA, int colsA, const int *B, int rowsB, int colsB, const int *addend)
{
	int rowsC = rowsA;
	int colsC = colsB;

	int *Ad = cuda_malloc_matrix_int(rowsA, colsA);
	int *Bd = cuda_malloc_matrix_int(rowsB, colsB);
	int *resultd = cuda_malloc_matrix_int(rowsC, colsC);

	cudaMemcpyToSymbol(ADDEND, addend, rowsC * colsC * sizeof(int));

	cudaMemcpy(Ad, A, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(rowsC, colsC);
	int sharedSize = (rowsA * colsA + rowsB * colsB) * sizeof(int); // Enough to fit A and B

	kernel_matrix_madd<<<1, threadsPerBlock, sharedSize>>>(resultd, Ad, rowsA, colsA, Bd, rowsB, colsB);

	cudaMemcpy(result, resultd, rowsC * colsC * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(resultd);
}

int main(int argc, char *argv[])
{
	if (argc < 5) {
		printf("Must supply rows and cols for each matrix\n");
		return 1;
	}

	int rowsA = atoi(argv[1]);
	int colsA = atoi(argv[2]);
	int rowsB = atoi(argv[3]);
	int colsB = atoi(argv[4]);
	int max = 200;

	if (rowsA > ROWS_MAX || rowsB > ROWS_MAX || colsA > COLS_MAX || colsB > COLS_MAX) {
		printf("Dimensions are too large. Max is %d x %d\n", ROWS_MAX, COLS_MAX);
		return 1;
	}

	if (argc > 5) {
		max = atoi(argv[5]);
	}

	if (colsA != rowsB) {
		printf("Incompatible dimensions. %d != %d\n", colsA, rowsB);
		return 1;
	}

	printf("RowsA: %d\nColsA: %d\nRowsB: %d\nColsB: %d\nMax: %d\n", rowsA, colsA, rowsB, colsB, max);

	int rowsC = rowsA;
	int colsC = colsB;
	int *multiplicand = malloc_matrix_int(rowsA, colsA);
	int *multiplier = malloc_matrix_int(rowsB, colsB);
	int *addend = malloc_matrix_int(rowsC, colsC);
	int *result = malloc_matrix_int(rowsC, colsC);

	generate_random_matrix(multiplicand, rowsA, colsA, max);
	generate_random_matrix(multiplier, rowsB, colsB, max);
	generate_random_matrix(addend, rowsC, colsC, max);

	print_matrix(multiplicand, rowsA, colsA, "A");
	print_matrix(multiplier, rowsB, colsB, "B");
	print_matrix(addend, rowsC, colsC, "C");

	accelerated_matrix_madd(result, multiplicand, rowsA, colsA, multiplier, rowsB, colsB, addend);

	print_matrix(result, rowsC, colsC, "A * B + C =");

	free(multiplicand);
	free(multiplier);
	free(addend);
	free(result);

	return 0;
}
