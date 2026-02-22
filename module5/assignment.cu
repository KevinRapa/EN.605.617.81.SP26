
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

#define ROWS_MAX 32
#define COLS_MAX 32

__constant__ int ADDEND[ROWS_MAX * COLS_MAX];

__global__ void kernel_matrix_madd(int *out, const int *a, int rowsA, int colsA, const int *b, int rowsB, int colsB)
{
}

void accelerated_matrix_madd(int *result, const int *A, int rowsA, int colsA, const int *B, int rowsB, int colsB, const int *addend)
{
	int rowsC = rowsB;
	int colsC = colsA;

	int *Ad = cuda_malloc_matrix_int(rowsA, colsA);
	int *Bd = cuda_malloc_matrix_int(rowsB, colsB);
	int *resultd = cuda_malloc_matrix_int(rowsC, colsC);

	cudaMemcpyToSymbol(ADDEND, addend, rowsC * colsC * sizeof(int));

	cudaMemcpy(Ad, A, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(rowsC, colsC);

	kernel_matrix_madd<<<1, threadsPerBlock, rowsA * colsA * sizeof(int)>>>(resultd, Ad, rowsA, colsA, Bd, rowsB, colsB);

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

	int *multiplicand = malloc_matrix_int(rowsA, colsA);
	int *multiplier = malloc_matrix_int(rowsB, colsB);
	int *addend = malloc_matrix_int(rowsB, colsA);
	int *result = malloc_matrix_int(rowsB, colsA);

	generate_random_matrix(multiplicand, rowsA, colsA, max);
	generate_random_matrix(multiplier, rowsB, colsB, max);
	generate_random_matrix(addend, rowsB, colsA, max);

	print_matrix(multiplicand, rowsA, colsA, "A");
	print_matrix(multiplier, rowsB, colsB, "B");
	print_matrix(addend, rowsB, colsA, "C");

	printf("Computing A * B + C");

	accelerated_matrix_madd(result, multiplicand, rowsA, colsA, multiplier, rowsB, colsB, addend);

	print_matrix(result, rowsB, colsA, "Result");

	free(multiplicand);
	free(multiplier);
	free(addend);
	free(result);

	return 0;
}
