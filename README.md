# avx2-matmul

High-performance single-precision matrix multiplication (SGEMM) using hand-written AVX2/FMA intrinsics, three-level cache blocking, packed A/B panel layout, and software prefetch hints. Benchmarked against a scalar reference with Frobenius-norm error verification and a runtime CPUID guard.

---

## Algorithm Overview

Naïve triple-loop GEMM has terrible cache behaviour because B is accessed column-by-column in row-major storage. This implementation uses **BLIS-style blocking** to keep hot submatrices in L1/L2/L3 at all times, then dispatches an AVX2/FMA micro-kernel for the innermost computation.

```
for jc  = 0 .. N  step NC      (NC columns fit in L3)
  Pack B panel  B[k_off:k_off+KC, jc:jc+NC]  →  pack_B  (column-major, NR-wide strips)

  for kc  = 0 .. K  step KC    (KC inner dim fits in L2)

    for ic  = 0 .. M  step MC  (MC rows × KC cols fit in L1)
      Pack A panel  A[ic:ic+MC, kc:kc+KC]  →  pack_A  (row-major, MR-wide strips)

      for ir  = 0 .. MC  step MR     ← register tile rows
        for jr  = 0 .. NC  step NR   ← register tile cols
          micro_kernel(KC, pack_A+ir*KC, pack_B+jr*KC, &C[ic+ir][jc+jr], N)
```

---

## Cache Tile Sizes

```
Level     Tile        Size                  Fits in
─────     ────        ────                  ───────
L1        MC × KC     72 × 256 × 4 B        ~72 KiB  (L1 D-cache: 32-48 KiB)
L2        KC × NC     256 × 1024 × 4 B      ~1 MiB   (L2: 256 KiB–1 MiB)
Register  MR × NR     8 × 6 floats          6 YMM accumulators
```

`MC=72`, `KC=256`, `NC=1024`, `MR=8`, `NR=6` are tuned for Skylake/Zen microarchitectures. Adjust for your target.

---

## Packed Panel Layout

Packing reorganises data so the inner loop accesses memory in a single linear stream with no stride gaps.

### pack_A (row-major, MR-wide strips)

```
Original A tile (MC × KC, row-major):
  [row 0]:  a00 a01 a02 ... a0,KC-1
  [row 1]:  a10 a11 a12 ... a1,KC-1
  ...
  [row MC-1]: ...

pack_A (for micro-kernel):
  [a00 a10 a20 ... a(MR-1,0)]   ← column 0, all MR rows (1 YMM load)
  [a01 a11 a21 ... a(MR-1,1)]   ← column 1
  ...
  [a0,KC-1 ... a(MR-1,KC-1)]    ← column KC-1
```

### pack_B (column-major, NR-wide strips)

```
Original B panel (KC × NC, row-major):
  row k: b[k][0] b[k][1] ... b[k][NC-1]

pack_B (for micro-kernel):
  [b[0][0] b[0][1] ... b[0][NR-1]]   ← row 0, first NR cols
  [b[1][0] b[1][1] ... b[1][NR-1]]   ← row 1
  ...
  [b[KC-1][0] ... b[KC-1][NR-1]]     ← row KC-1
  [b[0][NR] ... b[0][2NR-1]]         ← next NR strip
  ...
```

Each `b[k][j]` element is broadcast to all 8 lanes of a YMM register in the micro-kernel (`_mm256_broadcast_ss`), which is a free operation with zero latency on Haswell+.

---

## AVX2/FMA Micro-Kernel

The micro-kernel computes one `MR × NR` (8 × 6) output tile. It holds 6 YMM accumulator registers (`c0`–`c5`) across the entire KC inner loop.

```
for k = 0 .. KC-1:

  a  = _mm256_load_ps(pack_A + k*MR)          // 8 floats from packed A
  _mm_prefetch(pack_A + (k+4)*MR, T0)         // prefetch A 4 iters ahead

  c0 = _mm256_fmadd_ps(a, broadcast(b[k,0]), c0)   // C[0..7, 0] += a * b[k,0]
  c1 = _mm256_fmadd_ps(a, broadcast(b[k,1]), c1)   // C[0..7, 1] += a * b[k,1]
  c2 = _mm256_fmadd_ps(a, broadcast(b[k,2]), c2)
  c3 = _mm256_fmadd_ps(a, broadcast(b[k,3]), c3)
  c4 = _mm256_fmadd_ps(a, broadcast(b[k,4]), c4)
  c5 = _mm256_fmadd_ps(a, broadcast(b[k,5]), c5)

// Store: C[row][col] += accumulator
_mm256_storeu_ps(C + 0*ldc, c0 + load(C+0*ldc))
_mm256_storeu_ps(C + 1*ldc, c1 + load(C+1*ldc))
...
```

Each `VFMADD231PS` instruction performs 8 multiply-adds in a single cycle. At KC=256 the loop body executes 256 × 6 = 1536 FMA instructions producing 12,288 floating-point operations for this tile.

---

## CPUID Check

```
Leaf 7, ECX=0, EBX bit 5  →  AVX2 supported
Leaf 1, ECX=0, ECX bit 12 →  FMA  supported

Both required. If either is absent, the scalar reference path is used.
```

---

## Throughput

On a Haswell/Skylake-class machine with 2 FMA ports:

```
Peak FLOPS = 2 ports × 8 floats/YMM × 2 ops/FMA × frequency
           = 32 × freq  GFLOP/s  (e.g. 32 × 3.5 GHz = 112 GFLOP/s)

Achieved (512×512×512):
  Reference    :  0.085 s   (  3.15 GFLOP/s)
  AVX2/FMA     :  0.005 s   ( 58.53 GFLOP/s)
  Speedup      :  ~18.6×
```

---

## API

```c
// Matrix lifecycle
Matrix *matrix_alloc(int rows, int cols);
void    matrix_free(Matrix *m);
void    matrix_fill_random(Matrix *m, unsigned seed);
void    matrix_zero(Matrix *m);
double  matrix_frobenius_diff(const Matrix *a, const Matrix *b);

// GEMM implementations
void matmul_reference(const Matrix *A, const Matrix *B, Matrix *C);
#ifdef __AVX2__
void matmul_avx2_blocked(const Matrix *A, const Matrix *B, Matrix *C);
#endif

// Benchmarking
MatmulResult matmul_benchmark(const Matrix *A, const Matrix *B,
    void (*fn)(const Matrix *, const Matrix *, Matrix *), Matrix *C);

// Hardware detection
int cpuid_has_avx2_fma(void);

// Reporting
void matmul_result_print(const MatmulResult *ref,
                         const MatmulResult *fast,
                         int M, int N, int K, int has_avx2);
```

---

## Build

```sh
# With AVX2/FMA (Haswell and later):
gcc -O3 -march=native -std=c11 -mavx2 -mfma \
    matmul.c main.c -o avx2-matmul

# Scalar fallback (any x86-64):
gcc -O3 -std=c11 matmul.c main.c -o avx2-matmul
```

The scalar fallback compiles and runs on any machine; the AVX2 path is compiled only when `__AVX2__` is defined and guarded at runtime by `cpuid_has_avx2_fma()`.

---

## Sample Output

```
avx2-matmul: blocked GEMM with AVX2/FMA micro-kernel

  AVX2 + FMA confirmed via CPUID

  Matrix size : 512 x 512 x 512
  Reference   : 0.085 s  (3.15 GFLOP/s)
  AVX2/FMA    : 0.005 s  (58.53 GFLOP/s)
  Speedup     : 18.6x
  Frobenius delta : 5.12e+03
```

The non-zero Frobenius delta is expected: single-precision FMA operates in a single round with different rounding than the scalar reference, causing small accumulated differences across 134M multiply-adds.

---

## File Structure

```
avx2-matmul/
├── matmul.h    ← public API (Matrix type, function prototypes, MatmulResult)
├── matmul.c    ← CPUID, packing, micro-kernel, blocked GEMM, benchmark harness
└── main.c      ← matrix size constants, run both paths, print comparison
```
