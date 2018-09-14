#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

typedef void (*gemm_fn_t)(int, int, int,
                          const double *, int,
                          const double *, int,
                          double *, int);

void optimised_gemm(int, int, int,
                    const double *, int,
                    const double *, int,
                    double *, int);

void basic_gemm(int, int, int,
                const double *, int,
                const double *, int,
                double *, int);

/*
 * Print entries of a matrix.
 * if num_entries is < 0, print all entries, otherwise just print the
 * top num_entries x num_entries corner.
 */
static void print_matrix(int m, int n, const double *a, int lda,
                         int num_entries)
{
    int i, j;
    if (num_entries >= 0) {
        n = n < num_entries ? n : num_entries;
        m = m < num_entries ? m : num_entries;
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%.5g ", a[j*lda + i]);
        }
        printf("\n");
    }
}

static void zero_matrix(int m, int n, double *a, int lda)
{
    int i, j;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            a[j*lda + i] = 0.0;
        }
    }    
}

static void random_matrix(int m, int n, double *a, int lda)
{
    int i, j;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            a[j*lda + i] = drand48();
        }
    }
}

static double diff_time(struct timespec end, struct timespec start)
{
    double secs = (double)end.tv_sec - (double)start.tv_sec;
    double nsecs = (double)end.tv_nsec - (double)start.tv_nsec;
    return secs + 1e-9*nsecs;
}

/*
 * Check that a matrix matches the result of multiplication with
 * "basic" implementation.
 * m, n, k: size of matrices to check.
 * C has rank m x n (m rows, n columns)
 * A has rank m x k
 * B has rank k x n
 * gemm: function pointer to gemm implementation.
 * Returns 1 if the check failed, 0 if it passed.
 */
static int check(int m, int n, int k, gemm_fn_t gemm, double *maxdiff)
{
    double *a = NULL;
    double *b = NULL;
    double *copt = NULL;
    double *cbasic = NULL;
    *maxdiff = -1;
    int i, j;
    int lda, ldb, ldc;

    a = malloc(m*k * sizeof(double));
    b = malloc(k*n * sizeof(double));
    copt = malloc(m*n * sizeof(double));
    cbasic = malloc(m*n * sizeof(double));

    lda = m;
    ldb = k;
    ldc = m;

    random_matrix(m, k, a, lda);
    random_matrix(k, n, b, ldb);
    zero_matrix(m, n, copt, ldc);
    zero_matrix(m, n, cbasic, ldc);

    basic_gemm(m, n, k, a, lda, b, ldb, cbasic, ldc);
    gemm(m, n, k, a, lda, b, ldb, copt, ldc);

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            double diff = fabs(copt[j*ldc + i] - cbasic[j*ldc + i]);
            if (diff != diff) {
                *maxdiff = diff;
                goto done;
            } else {
                *maxdiff = fmax(*maxdiff, diff);
            }
        }
    }
done:
    free(a);
    free(b);
    free(copt);
    free(cbasic);
    return (*maxdiff > 1e-3) || (*maxdiff != *maxdiff);
}

/*
 * Benchmark the provided gemm implementation.
 * m, n, k: matrix sizes C[m, n] = C[m, n] + A[m, k]*B[k, n]
 * gemm: Function pointer to gemm implementation
 * prints:
 *  m n k TIME FLOP FLOP/s
 */
static void bench(int m, int n, int k, gemm_fn_t gemm)
{
    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    struct timespec start, end;
    double time, flop;
    int repeats, i;
    int lda, ldb, ldc;

    a = malloc(m*k * sizeof(double));
    b = malloc(k*n * sizeof(double));
    c = malloc(m*n * sizeof(double));

    lda = m;
    ldb = k;
    ldc = m;

    random_matrix(m, k, a, lda);
    random_matrix(k, n, b, ldb);
    zero_matrix(m, n, c, ldc);

    flop = 2.0*(double)m*(double)n*(double)k;
    time = DBL_MAX;
    
    if (m*n < 10000) {
        /* For small matrices, run in a loop, to help with timing variability. */
        repeats = 50;
    } else {
        repeats = 2;
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for (i = 0; i < repeats; i++) {
        gemm(m, n, k,
             (const double *)a, lda,
             (const double *)b, ldb,
             c, ldc);
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    time = diff_time(end, start) / repeats;
    printf("%d %d %d %g %g\n", m, n, k, time, flop);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv)
{
    int m, n, k;
    if (argc != 5) {
        fprintf(stderr, "Invalid arguments.\n");
        fprintf(stderr, "Usage: %s M N K mode\n", argv[0]);
        fprintf(stderr, "Where M, N, and K are the dimensions of the problem.\n");
        fprintf(stderr, "'mode' is one of BENCH or CHECK\n");
        return 1;
    }

    /* A is m x k; B is k x n; C is m x n. */
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

    if (!strcmp(argv[4], "BENCH")) {
        bench(m, n, k, &optimised_gemm);
    } else if (!strcmp(argv[4], "CHECK")) {
        double maxdiff;
        int val = check(m, n, k, &optimised_gemm, &maxdiff);
        if (val) {
            fprintf(stderr, "CHECK FAILED, maximum entry difference %g\n", maxdiff);
        } else {
            printf("CHECK SUCCEEDED\n");
        }
    } else {
        fprintf(stderr, "Unrecognised mode %s, should be BENCH or CHECK\n", argv[4]);
        return 1;
    }
    return 0;
}
