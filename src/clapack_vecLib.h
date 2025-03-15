#define FFI_SCOPE "Rindow\\OpenBLAS\\FFI"

typedef int32_t                     __CLPK_integer;
typedef float                       __CLPK_real;
typedef double                      __CLPK_doublereal;
typedef __CLPK_integer              lapack_int;

int sgesvd_(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_real *__a, __CLPK_integer *__lda,
        __CLPK_real *__s, __CLPK_real *__u, __CLPK_integer *__ldu,
        __CLPK_real *__vt, __CLPK_integer *__ldvt, __CLPK_real *__work,
        __CLPK_integer *__lwork,
        __CLPK_integer *__info);

int dgesvd_(char *__jobu, char *__jobvt, __CLPK_integer *__m,
        __CLPK_integer *__n, __CLPK_doublereal *__a, __CLPK_integer *__lda,
        __CLPK_doublereal *__s, __CLPK_doublereal *__u, __CLPK_integer *__ldu,
        __CLPK_doublereal *__vt, __CLPK_integer *__ldvt,
        __CLPK_doublereal *__work, __CLPK_integer *__lwork,
        __CLPK_integer *__info);
