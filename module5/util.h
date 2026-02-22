
#ifndef UTIL_H
#define UTIL_H

void generate_random_matrix(int *out, int rows, int cols, int max);
void print_matrix(const int *in, int rows, int cols, const char *name);
int *malloc_matrix_int(int rows, int cols);
int *cuda_malloc_matrix_int(int rows, int cols);

#endif
