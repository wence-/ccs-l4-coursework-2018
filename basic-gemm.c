/* Compute C = C + A*B
 *
 * C has rank m x n (m rows, n columns)
 * A has rank m x k
 * B has rank k x n
 * ldX is the leading dimension of the respective matrix.
 *
 * All matrices are stored in column major format.
 * That is, for row index i, column index j, and leading dimension ldX,
 * the correct entry is at (ldX*i + j).
 */
void basic_gemm(int m, int n, int k,
                const double *a, int lda,
                const double *b, int ldb,
                double *c, int ldc)
{
    int i, j, p;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            /* c[i, j] <-  c[i, j] + a[i, p] * b[p, j]
             * i.e. the dot product of the ith row of a with the jth
             * column of b. */
            for (p = 0; p < k; p++) {
                c[j*ldc + i] = c[j*ldc + i] + a[p*lda + i] * b[j*ldb + p];
            }
        }
    }
}
