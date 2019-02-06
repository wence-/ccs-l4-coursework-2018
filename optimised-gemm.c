#include <stdlib.h>

/*
 * This implementation follows in part the UlmBLAS tutorial from
 * http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
 */

#define MC 1024          /* Height of B block row */
#define KC 256          /* Width of A block column */
#define NC 4096         /* Length of B block row */
#define MR 4            /* Rows of output matrix updated at once */
#define NR 8            /* Columns of output matrix updated at once */

static void pack_A_full(int k,
                        const double * restrict A, int lda,
                        double * restrict buffer)
{
  int i, j;

  for (j = 0; j < k; ++j)
    for (i = 0; i < MR; ++i)
      buffer[i + j*MR] = A[i + j*lda];
}

static void pack_A(int m, int k,
                   const double * restrict A, int lda,
                   double * restrict buffer)
{
  int i, j;
  int mp  = m / MR;
  int _mr = m % MR;

  for (i = 0; i < mp; ++i)
    /* Pack A, in row strips MR x k, column major order. */
    pack_A_full(k, &A[i*MR], lda, &buffer[i*k*MR]);
  if (_mr) {
    /* Cleanup code for non-full tile */
    for (j = 0; j < k; ++j) {
      for (i = 0; i < _mr; ++i)
        buffer[i + j*MR + mp*k*MR] = A[mp*MR + j*lda + i];
      for (i = _mr; i < MR; ++i)
        buffer[i + j*MR + mp*k*MR] = 0.0;
    }
  }
}

static void pack_B_full(int k,
                        const double * restrict B, int ldb,
                        double * restrict buffer)
{
  int i, j;

  for (i = 0; i < k; ++i)
    for (j = 0; j < NR; ++j)
      buffer[j + i*NR] = B[j*ldb + i];
}

static void pack_B(int k, int n,
                   const double * restrict B, int ldb,
                   double * restrict buffer)
{
  int i, j;
  int np  = n / NR;
  int _nr = n % NR;

  for (j = 0; j < np; ++j)
    /* Pack B, in column strips kc x NR, row major order. */
    pack_B_full(k, &B[j*NR*ldb], ldb, &buffer[j*k*NR]);
  if (_nr) {
    /* Cleanup code for non full tile. */
    for (i = 0; i < k; ++i) {
      for (j = 0; j < _nr; ++j)
        buffer[j + i*NR + np*k*NR] = B[j*ldb + i + np*NR*ldb];
      for (j = _nr; j < NR; ++j)
        buffer[j + i*NR + np*k*NR] = 0.0;
    }
  }
}

/* Seems to be better not to inline this for ICC. */
#ifdef __ICC
__attribute__((noinline))
#endif
static void micro_kernel(int kc,
                         const double * restrict A,
                         const double * restrict B,
                         double * restrict AB)
{
  /* Compute a little MR x NR output block in C. */
  int i, j, l;

  /* For every "block" column */
  for (l = 0; l < kc; ++l)
    /* ICC seems to do a good job unrolling this and producing good
       code. Clang and GCC need more work. */
#pragma unroll
    for (j = 0; j < NR; ++j)
#pragma omp simd
      for (i = 0; i < MR; ++i)
        /* Multiply row of A into column of B. */
        AB[i + j*MR] += A[i + l*MR] * B[j + l*NR];
}

static void macro_kernel(int mc, int nc, int kc,
                         double * restrict _A,
                         double * restrict _B,
                         double * restrict C, int ldc)
{
  int i, j;
  int mp = (mc+MR-1) / MR;
  int np = (nc+NR-1) / NR;

  int _mr = mc % MR;
  int _nr = nc % NR;


  for (j = 0; j < np; ++j) {
    /* Only the last iteration might not be a full tile */
    int nr = (j != np-1 || _nr == 0) ? NR : _nr;

    for (i = 0; i < mp; ++i) {
      int k, l;
      int mr = (i != mp-1 || _mr == 0) ? MR : _mr;
      double _C[MR*NR] __attribute__((aligned(64))) = {0};

      /* Multiply into temporary */
      micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR], _C);

      /* Update output matrix. Do this separately for better locality
       * in the hot inner loop: AB can live in registers in the micro
       * kernel. */
      for (k = 0; k < nr; ++k)
        for (l = 0; l < mr; ++l)
          C[i*MR + j*NR*ldc + k*ldc + l] += _C[k*MR + l];
    }
  }
}

void optimised_gemm(int m, int n, int k,
                    const double * restrict A, int lda,
                    const double * restrict B, int ldb,
                    double * restrict C, int ldc)
{
  /*
   * Local buffers for storing panels from A, and B.
   */
  double *_A;
  double *_B;

  int i, j, l;

  /* Number of full blocks */
  int mb = (m+MC-1) / MC;
  int nb = (n+NC-1) / NC;
  int kb = (k+KC-1) / KC;

  /* Clean up tiles */  
  int _mc = m % MC;  
  int _nc = n % NC;
  int _kc = k % KC;

  posix_memalign((void**)&_A, 64, sizeof(*_A)*MC*KC);
  posix_memalign((void**)&_B, 64, sizeof(*_B)*KC*NC);

  for (j = 0; j < nb; ++j) {
    /* Only the last iteration might not be a full tile */
    int nc = (j != nb-1 || _nc == 0) ? NC : _nc;

    for (l = 0; l < kb; ++l) {
      /* Only the last iteration might not be a full tile */
      int kc = (l != kb-1 || _kc == 0) ? KC : _kc;

      /* Pack kc x nc long thin row of B */
      pack_B(kc, nc, &B[l*KC + j*NC*ldb], ldb, _B);

      for (i = 0; i < mb; ++i) {
        /* Only the last iteration might not be a full tile */
        int mc = (i != mb-1 || _mc == 0) ? MC : _mc;

        /* Pack mc x kc tall thin column of A */
        pack_A(mc, kc, &A[i*MC + l*KC*lda], lda, _A);

        macro_kernel(mc, nc, kc, _A, _B, &C[i*MC + j*NC*ldc], ldc);
      }
    }
  }
  free(_A);
  free(_B);
}
