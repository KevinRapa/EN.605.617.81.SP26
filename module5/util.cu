
#include <stdlib.h>
#include <stdio.h>

__host__ void generate_random_matrix(int *out, int rows, int cols, int max)
{
	for (int i = 0; i < rows * cols; i++) {
		out[i] = (int)rand() % max;
	}
}

__host__ void print_matrix(const int *in, int rows, int cols, const char *name)
{
	printf("Matrix %s:\n", name);

	for (int i = 0; i < rows; i++) {
		printf("[ ");

		for (int j = 0; j < cols; j++) {
			printf("%4d, ", in[i * cols +j]);
		}

		printf(" ]\n");
	}
}

__host__ int *malloc_matrix_int(int rows, int cols)
{
	return (int *)malloc(rows * cols * sizeof(int));
}

__host__ int *cuda_malloc_matrix_int(int rows, int cols)
{
	int *m = NULL;
	cudaMalloc(&m, rows * cols * sizeof(int));
	return m;
}

