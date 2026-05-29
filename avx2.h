#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

typedef struct {
    float *data;
    int    rows;
    int    cols;
} Matrix;

typedef struct {
    double elapsed_seconds;
    double gflops;
    double frobenius_error;
} MatmulResult;

int     cpuid_has_avx2_fma(void);

Matrix *matrix_alloc(int rows, int cols);
void    matrix_free(Matrix *m);
void    matrix_fill_random(Matrix *m, unsigned seed);
void    matrix_zero(Matrix *m);
double  matrix_frobenius_diff(const Matrix *a, const Matrix *b);

void matmul_reference(const Matrix *A, const Matrix *B, Matrix *C);

#ifdef __AVX2__
void matmul_avx2_blocked(const Matrix *A, const Matrix *B, Matrix *C);
#endif

MatmulResult matmul_benchmark(const Matrix *A, const Matrix *B,
                               void (*fn)(const Matrix *, const Matrix *, Matrix *),
                               Matrix *C);

void matmul_result_print(const MatmulResult *ref,
                         const MatmulResult *fast,
                         int M, int N, int K,
                         int has_avx2);

#endif
