void basic_gemm(int, int, int,
                const double *, int,
                const double *, int,
                double *, int);

/* Compute C = C + A*B
 *
 * C has rank m x n
 * A has rank m x k
 * B has rank k x n
 * ldX is the leading dimension of the respective matrix.
 *
 * All matrices are stored in column major format.
 */
void optimised_gemm(int m, int n, int k,
                    const double *a, int lda,
                    const double *b, int ldb,
                    double *c, int ldc)
{
    /* Default implementation just calls the basic routine. */
    return basic_gemm(m, n, k, a, lda, b, ldb, c, ldc);
}
