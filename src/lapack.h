#define FFI_SCOPE "Rindow\\OpenBLAS\\FFI"

typedef int32_t                     lapack_int;

void sgesvd_(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    float* A, lapack_int const* lda,
    float* S,
    float* U, lapack_int const* ldu,
    float* VT, lapack_int const* ldvt,
    float* work, lapack_int const* lwork,
    lapack_int* info
);

void dgesvd_(
    char const* jobu, char const* jobvt,
    lapack_int const* m, lapack_int const* n,
    double* A, lapack_int const* lda,
    double* S,
    double* U, lapack_int const* ldu,
    double* VT, lapack_int const* ldvt,
    double* work, lapack_int const* lwork,
    lapack_int* info
);
