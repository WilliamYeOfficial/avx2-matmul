#define _GNU_SOURCE
#include "matmul.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#define ALIGN32 32

#define MC   72
#define KC  256
#define NC  1024
#define MR   8
#define NR   6

int cpuid_has_avx2_fma(void)
{
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));
    int avx2 = (ebx >> 5) & 1;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(1), "c"(0));
    int fma = (ecx >> 12) & 1;
    return avx2 && fma;
#else
    return 0;
#endif
}

Matrix *matrix_alloc(int rows, int cols)
{
    Matrix *m = malloc(sizeof *m);
    m->rows   = rows;
    m->cols   = cols;
    m->data   = aligned_alloc(ALIGN32, sizeof(float) * (size_t)rows * (size_t)cols);
    return m;
}

void matrix_free(Matrix *m)
{
    if (m) { free(m->data); free(m); }
}

void matrix_fill_random(Matrix *m, unsigned seed)
{
    srand(seed);
    int n = m->rows * m->cols;
    for (int i = 0; i < n; ++i)
        m->data[i] = (float)(rand() % 200 - 100) * 0.01f;
}

void matrix_zero(Matrix *m)
{
    memset(m->data, 0, sizeof(float) * (size_t)m->rows * (size_t)m->cols);
}

double matrix_frobenius_diff(const Matrix *a, const Matrix *b)
{
    int n = a->rows * a->cols;
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a->data[i] - (double)b->data[i];
        s += d * d;
    }
    /* Manual sqrt via Newton-Raphson to avoid mandatory -lm linkage */
    if (s <= 0.0) return 0.0;
    double x = s;
    for (int i = 0; i < 60; ++i) x = 0.5 * (x + s / x);
    return x;
}

void matmul_reference(const Matrix *A, const Matrix *B, Matrix *C)
{
    int M = A->rows, K = A->cols, N = B->cols;
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A->data[i * K + k];
            for (int j = 0; j < N; ++j)
                C->data[i * N + j] += a * B->data[k * N + j];
        }
}

#ifdef __AVX2__

static float *g_pack_B = NULL;
static float *g_pack_A = NULL;

static void pack_b_panel(const float *B, int ldb, int kc, int nc)
{
    float *dst = g_pack_B;
    for (int j = 0; j < nc; j += NR) {
        int jb = (j + NR <= nc) ? NR : nc - j;
        for (int k = 0; k < kc; ++k) {
            for (int jj = 0; jj < jb; ++jj) *dst++ = B[k * ldb + j + jj];
            for (int jj = jb; jj < NR; ++jj) *dst++ = 0.0f;
        }
    }
}

static void pack_a_panel(const float *A, int lda, int mc, int kc)
{
    float *dst = g_pack_A;
    for (int i = 0; i < mc; i += MR) {
        int ib = (i + MR <= mc) ? MR : mc - i;
        for (int k = 0; k < kc; ++k) {
            for (int ii = 0; ii < ib; ++ii) *dst++ = A[(i + ii) * lda + k];
            for (int ii = ib; ii < MR; ++ii) *dst++ = 0.0f;
        }
    }
}

static void micro_kernel_scalar(int kc, int mb, int jb,
                                const float *pa, const float *pb,
                                float *C, int ldc)
{
    for (int i = 0; i < mb; ++i)
        for (int k = 0; k < kc; ++k)
            for (int j = 0; j < jb; ++j)
                C[i * ldc + j] += pa[i * kc + k] * pb[k * NR + j];
}

static void micro_kernel_avx2(int kc, const float *pa, const float *pb,
                               float *C, int ldc)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        __m256 a = _mm256_load_ps(pa + k * MR);
        _mm_prefetch((const char *)(pa + (k + 4) * MR), _MM_HINT_T0);
        c0 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 0), c0);
        c1 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 1), c1);
        c2 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 2), c2);
        c3 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 3), c3);
        c4 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 4), c4);
        c5 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(pb + k * NR + 5), c5);
    }

    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(c0, _mm256_loadu_ps(C + 0 * ldc)));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(c1, _mm256_loadu_ps(C + 1 * ldc)));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(c2, _mm256_loadu_ps(C + 2 * ldc)));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(c3, _mm256_loadu_ps(C + 3 * ldc)));
    _mm256_storeu_ps(C + 4 * ldc, _mm256_add_ps(c4, _mm256_loadu_ps(C + 4 * ldc)));
    _mm256_storeu_ps(C + 5 * ldc, _mm256_add_ps(c5, _mm256_loadu_ps(C + 5 * ldc)));
}

void matmul_avx2_blocked(const Matrix *A, const Matrix *B, Matrix *C)
{
    int M = A->rows, K = A->cols, N = B->cols;
    g_pack_B = aligned_alloc(ALIGN32, sizeof(float) * KC * (NC + NR));
    g_pack_A = aligned_alloc(ALIGN32, sizeof(float) * MC * (KC + MR));

    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC <= N) ? NC : N - jc;
        for (int kc_off = 0; kc_off < K; kc_off += KC) {
            int kc = (kc_off + KC <= K) ? KC : K - kc_off;
            pack_b_panel(B->data + kc_off * N + jc, N, kc, nc);
            for (int ic = 0; ic < M; ic += MC) {
                int mc = (ic + MC <= M) ? MC : M - ic;
                pack_a_panel(A->data + ic * K + kc_off, K, mc, kc);
                for (int ir = 0; ir < mc; ir += MR) {
                    int mb = (ir + MR <= mc) ? MR : mc - ir;
                    for (int jr = 0; jr < nc; jr += NR) {
                        const float *pa  = g_pack_A + ir * kc;
                        const float *pb  = g_pack_B + jr * kc;
                        float       *Cij = C->data + (ic + ir) * N + (jc + jr);
                        if (mb == MR)
                            micro_kernel_avx2(kc, pa, pb, Cij, N);
                        else
                            micro_kernel_scalar(kc, mb,
                                (jr + NR <= nc) ? NR : nc - jr,
                                pa, pb, Cij, N);
                    }
                }
            }
        }
    }

    free(g_pack_B); g_pack_B = NULL;
    free(g_pack_A); g_pack_A = NULL;
}
#endif

static double timespec_elapsed(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
}

MatmulResult matmul_benchmark(const Matrix *A, const Matrix *B,
                               void (*fn)(const Matrix *, const Matrix *, Matrix *),
                               Matrix *C)
{
    matrix_zero(C);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fn(A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = timespec_elapsed(t0, t1);
    double ops     = 2.0 * A->rows * B->cols * A->cols;

    return (MatmulResult){
        .elapsed_seconds  = elapsed,
        .gflops           = ops / elapsed / 1e9,
        .frobenius_error  = 0.0,
    };
}

void matmul_result_print(const MatmulResult *ref, const MatmulResult *fast,
                         int M, int N, int K, int has_avx2)
{
    printf("avx2-matmul: blocked GEMM with AVX2/FMA micro-kernel\n\n");
    if (!has_avx2)
        printf("  WARNING: AVX2/FMA not detected — scalar fallback used\n\n");
    else
        printf("  AVX2 + FMA confirmed via CPUID\n\n");

    printf("  Matrix size : %d x %d x %d\n", M, N, K);
    printf("  Reference   : %.3f s  (%.2f GFLOP/s)\n",
           ref->elapsed_seconds, ref->gflops);

    if (has_avx2 && fast) {
        printf("  AVX2/FMA    : %.3f s  (%.2f GFLOP/s)\n",
               fast->elapsed_seconds, fast->gflops);
        printf("  Speedup     : %.1fx\n",
               ref->elapsed_seconds / fast->elapsed_seconds);
        printf("  Frobenius delta : %.2e\n", fast->frobenius_error);
    }
}
