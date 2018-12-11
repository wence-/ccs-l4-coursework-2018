/* Compute C = C + A*B
 *
 * C has rank m x n
 * A has rank m x k
 * B has rank k x n
 * ldX is the leading dimension of the respective matrix.
 *
 * All matrices are stored in column major format.
 */

#ifdef __ICC
#define __builtin_assume(a) __assume(a)
#define __builtin_assume_aligned(a, b) ( __assume_aligned(a, b))
#elif __clang__
#define __builtin_assume_aligned(a, b) (a = __builtin_assume_aligned(a, b))
#else
#define __builtin_assume(a)
#endif

#define MC 256                  /* Height of B block row */
#define KC 256                  /* Width of A block column */
#define NC 4096                 /* Length of B block row */
#define MR 4            /* Rows of output matrix updated at once */
#define NR 8            /* Columns of output matrix updated at once */


/*
 * Local buffers for storing panels from A, B and C, not thread safe.
 * But this is not important.
 */
static double _A[MC*KC] __attribute__((aligned(64)));
static double _B[KC*NC] __attribute__((aligned(64)));
static double _C[MR*NR] __attribute__((aligned(64)));

static void
pack_MRxk(int k, const double * restrict A, int lda, double * restrict buffer)
{
  int i, j;

  for (j=0; j<k; ++j) {
    for (i=0; i<MR; ++i) {
      buffer[i + j*MR] = A[i + j * lda];
    }
  }
}

static void
pack_A(int mc, int kc, const double * restrict A, int lda, double * restrict buffer)
{
  int mp  = mc / MR;
  int _mr = mc % MR;

  int i, j;

  for (i=0; i<mp; ++i) {
    /* Pack A, in row strips MR x kc, column major order. */
    pack_MRxk(kc, &A[i*MR], lda, &buffer[i*kc*MR]);
  }
  if (_mr>0) {
    /* Cleanup code for non-full tile */
    for (j=0; j<kc; ++j) {
      for (i=0; i<MR; ++i) {
        buffer[i + j*MR + mp*kc * MR] = i < _mr ? A[mp * MR + j * lda + i] : 0.0;
      }
    }
  }
}

static void
pack_kxNR(int k, const double * restrict B, int ldb, double * restrict buffer)
{
  int i, j;

  for (i=0; i<k; ++i) {
    for (j=0; j<NR; ++j) {
      buffer[j + i * NR] = B[j*ldb + i];
    }
  }
}

static void
pack_B(int kc, int nc, const double * restrict B, int ldb, double * restrict buffer)
{
  int np  = nc / NR;
  int _nr = nc % NR;

  int i, j;

  for (j=0; j<np; ++j) {
    /* Pack A, in column strips kc x NR, row major order. */
    pack_kxNR(kc, &B[j*NR*ldb], ldb, &buffer[j*kc*NR]);
  }
  if (_nr>0) {
    /* Cleanup code for non full tile. */
    for (i=0; i<kc; ++i) {
      for (j=0; j<NR; ++j) {
        buffer[j + i*NR + np*kc*NR] = j < _nr ? B[j*ldb + i + np*NR*ldb] : 0.0;
      }
    }
  }
}

__attribute__((noinline))
static void
dgemm_micro_kernel(int kc,
                   const double * restrict A, const double * restrict B,
                   double * restrict C, int ldc)
{
  /* Compute a little MR x NR output block in C. */
  double AB[MR*NR] __attribute__((aligned(64)));

  int i, j, l;
  /* Zero output buffer. */
  for (l=0; l<MR*NR; ++l) {
    AB[l] = 0;
  }

  /* For every "block" column */
  for (l=0; l<kc; ++l) {
#pragma clang loop vectorize(enable)
    for (j=0; j<NR; ++j) {
#pragma omp simd
      for (i=0; i<MR; ++i) {
        /* Multiply row of A into column of B. */
        AB[i+j*MR] += A[i + l*MR]*B[j + l*NR];
      }
    }
  }

  /* Update output matrix. */
  for (j=0; j<NR; ++j) {
    for (i=0; i<MR; ++i) {
      C[i + j*ldc] += AB[i+j*MR];
    }
  }
}

static void dgeaxpy(int m,
                    int n,
                    const double * restrict X,
                    int ldx,
                    double * restrict Y,
                    int ldy)
{
  int i, j;

  /* Y <- Y + X */
  for (j=0; j<n; ++j) {
    for (i=0; i<m; ++i) {
      Y[i + j*ldy] += X[i + j*ldx];
    }
  }
}

static void
dgemm_macro_kernel(int mc,
                   int nc,
                   int kc,
                   double * restrict C,
                   int ldc)
{
  int mp = (mc+MR-1) / MR;
  int np = (nc+NR-1) / NR;

  int _mr = mc % MR;
  int _nr = nc % NR;

  int mr, nr;
  int i, j;

  for (j=0; j<np; ++j) {
    nr    = (j!=np-1 || _nr==0) ? NR : _nr;

    for (i=0; i<mp; ++i) {
      mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

      if (mr==MR && nr==NR) {
        /* Full panel */
        dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR], &C[i*MR + j*NR*ldc], ldc);
      } else {
        /* Cleanup panel */
        int k;
        for (k = 0; k < MR*NR; k++) {
          _C[k] = 0;
        }
        /* Multiply into temporary */
        dgemm_micro_kernel(kc, &_A[i*kc*MR], &_B[j*kc*NR], _C, MR);
        /* And update C */
        dgeaxpy(mr, nr, _C, MR, &C[i*MR + j*NR*ldc], ldc);
      }
    }
  }
}

void optimised_gemm(int m,
                    int n,
                    int k,
                    const double * restrict A,
                    int lda,
                    const double * restrict B,
                    int ldb,
                    double * restrict C,
                    int ldc)
{
  /*
   * #ifndef __ICC
   * return cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
   *                    m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
   * #endif
   */
  
  int mb = (m+MC-1) / MC;       /* Number of full blocks */
  int nb = (n+NC-1) / NC;
  int kb = (k+KC-1) / KC;

  int _mc = m % MC;             /* Clean up tiles */
  int _nc = n % NC;
  int _kc = k % KC;

  int mc, nc, kc;
  int i, j, l;

  for (j=0; j<nb; ++j) {
    nc = (j!=nb-1 || _nc==0) ? NC : _nc;

    for (l=0; l<kb; ++l) {
      kc = (l!=kb-1 || _kc==0) ? KC : _kc;

      /* Pack kc x nc long thin row of B */
      pack_B(kc, nc, &B[l*KC + j*NC*ldb], ldb, _B);

      for (i=0; i<mb; ++i) {
        mc = (i!=mb-1 || _mc==0) ? MC : _mc;

        /* Pack mc x kc tall thin column of A */
        pack_A(mc, kc, &A[i*MC + l*KC*lda], lda, _A);

        dgemm_macro_kernel(mc, nc, kc, &C[i*MC + j*NC*ldc], ldc);
      }
    }
  }
}

