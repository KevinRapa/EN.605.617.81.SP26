/**
 * Author: Kevin Rapa
 * Description: Demonstrates a kernel that mutliplies 2 matrices (up to size 32x32) and then adds a third one.
 *              Uses constant memory by storing the addend matrix in constant memory.
 *              Uses shared memory in kernel to transpose matrix B
 *              USes register memory in kernel to hold a running sum
 */

#include <stdio.h>
#include <stdlib.h>
#include "util.h"

// Maximums are defined by the maximum constant block size.
#define ROWS_MAX 32
#define COLS_MAX 32

// Holds the matrix we will add to the matrix product.
__constant__ int ADDEND[ROWS_MAX * COLS_MAX];

// Multiplies two matrices and then adds a matrix from constant memory.
__global__ void kernel_matrix_madd(int *out, const int *a, int rowsA, int colsA, const int *b, int rowsB, int colsB)
{
	// matrixCache holds matrix A and the transposed matrix B
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

	for (int i = 0; i < colsA; i++) {
		result += matrixCache[(threadIdx.x * colsA) + i] * matrixCache[(rowsA * colsA) + (threadIdx.y * colsB) + i];
	}

	// A lot of trial and error here to be honest... I got to point where I could get the transpose answer in the output buffer.
	// So since I am too burnt out, I'll just copy the transpose into a shared buffer and then transpose it into the final output.
	extern __shared__ int outTranspose[];

	outTranspose[threadIdx.y * blockDim.x + threadIdx.x] = result + ADDEND[threadIdx.y * blockDim.x + threadIdx.x];
	
	__syncthreads();

	out[threadIdx.y * blockDim.x + threadIdx.x] = outTranspose[threadIdx.x * blockDim.y + threadIdx.y];
}

float accelerated_matrix_madd(int *result, const int *A, int rowsA, int colsA, const int *B, int rowsB, int colsB, const int *addend)
{
	int rowsC = rowsA;
	int colsC = colsB;
	cudaEvent_t start, end;
	float timeElapsed = 0;

	int *Ad = cuda_malloc_matrix_int(rowsA, colsA);
	int *Bd = cuda_malloc_matrix_int(rowsB, colsB);
	int *resultd = cuda_malloc_matrix_int(rowsC, colsC);

	cudaMemcpyToSymbol(ADDEND, addend, rowsC * colsC * sizeof(int));

	cudaMemcpy(Ad, A, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(rowsC, colsC);
	int sharedSize = (rowsA * colsA + rowsB * colsB) * sizeof(int); // Enough to fit A and B

	start = get_time();

	kernel_matrix_madd<<<1, threadsPerBlock, sharedSize>>>(resultd, Ad, rowsA, colsA, Bd, rowsB, colsB);

	end = get_time();
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&timeElapsed, start, end);

	cudaMemcpy(result, resultd, rowsC * colsC * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(resultd);

	return timeElapsed;
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
	float timeElapsed;

	srand(time(NULL));

	generate_random_matrix(multiplicand, rowsA, colsA, max);
	generate_random_matrix(multiplier, rowsB, colsB, max);
	generate_random_matrix(addend, rowsC, colsC, max);

	print_matrix(multiplicand, rowsA, colsA, "A");
	print_matrix(multiplier, rowsB, colsB, "B");
	print_matrix(addend, rowsC, colsC, "C");

	timeElapsed = accelerated_matrix_madd(result, multiplicand, rowsA, colsA, multiplier, rowsB, colsB, addend);

	print_matrix(result, rowsC, colsC, "A * B + C =");

	free(multiplicand);
	free(multiplier);
	free(addend);
	free(result);

	fprintf(stderr, "Time elapsed for sizes [%2d,%2d] * [%2d,%2d] + [%2d,%2d]: %f ms\n", rowsA, colsA, rowsB, colsB, rowsC, colsC, timeElapsed);

	return 0;
}
