#include "matmul.h"
#include <stdio.h>

#define M 512
#define N 512
#define K 512

int main(void)
{
    int has_avx2 = cpuid_has_avx2_fma();

    Matrix *A  = matrix_alloc(M, K);
    Matrix *B  = matrix_alloc(K, N);
    Matrix *C1 = matrix_alloc(M, N);
    Matrix *C2 = matrix_alloc(M, N);

    matrix_fill_random(A, 42);
    matrix_fill_random(B, 43);

    MatmulResult ref_r = matmul_benchmark(A, B, matmul_reference, C1);
    MatmulResult avx_r = {0};

#ifdef __AVX2__
    if (has_avx2) {
        avx_r = matmul_benchmark(A, B, matmul_avx2_blocked, C2);
        avx_r.frobenius_error = matrix_frobenius_diff(C1, C2);
    }
#endif

    matmul_result_print(&ref_r, has_avx2 ? &avx_r : NULL, M, N, K, has_avx2);

    matrix_free(A); matrix_free(B);
    matrix_free(C1); matrix_free(C2);
    return 0;
}
