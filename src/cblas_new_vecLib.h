#define FFI_SCOPE "Rindow\\OpenBLAS\\FFI"

typedef int32_t                     __LAPACK_int;
typedef struct _openblas_complex_float { float real, imag; } openblas_complex_float;
typedef struct _openblas_complex_double { double real, imag; } openblas_complex_double;

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114};
enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};

// MARK: Auxiliary Routines and Extensions

__LAPACK_int cblas_errprn(__LAPACK_int ierr,  __LAPACK_int info, char * form, ...);
void cblas_xerbla(__LAPACK_int p, char * rout, char * form, ...);
/* Apple extensions to the BLAS interface. */

/* These routines perform linear operations (scalar multiplication and addition);
 * on matrices, with optional transposition.  Specifically, the operation is:
 *
 *      C = alpha * A + beta * B
 *
 * where A and B are optionally transposed as indicated by the value of transA
 * and transB.  This is a surprisingly useful operation; although its function
 * is fairly trivial, efficient implementation has enough subtlety to justify
 * a library interface.
 *
 * As an added convenience, this function supports in-place operation for
 * square matrices; in-place operation for non-square matrices in the face of
 * transposition is a subtle problem outside the scope of this interface.
 * In-place operation is only supported if the leading dimensions match as well
 * as the pointers.  If C overlaps A or B but does not have equal leading
 * dimension, or does not exactly match the source that it overlaps, the
 * behavior of this function is undefined.
 *
 * If alpha or beta is zero, then A (or B, respectively); is ignored entirely,
 * meaning that the memory is not accessed and the data does not contribute
 * to the result (meaning you can pass B == NULL if beta is zero);.
 *
 * Note that m and n are the number of rows and columns of C, respectively.
 * If either A or B is transposed, then they are interpreted as n x m matrices.
 */

extern void appleblas_sgeadd(const enum CBLAS_ORDER ORDER,
                             const enum CBLAS_TRANSPOSE TRANSA,
                             const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                             const float ALPHA, const float * A, const __LAPACK_int LDA,
                             const float BETA, const float * B, const __LAPACK_int LDB, float * C,
                             const __LAPACK_int LDC);
extern void appleblas_dgeadd(const enum CBLAS_ORDER ORDER,
                             const enum CBLAS_TRANSPOSE TRANSA,
                             const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                             const double ALPHA, const double * A, const __LAPACK_int LDA,
                             const double BETA, const double * B, const __LAPACK_int LDB, double * C,
                             const __LAPACK_int LDC);
/* The BLAS standard defines a function, cblas_xerbla( );, and suggests that
 * programs provide their own implementation in order to override default
 * error handling.  This scheme is incompatible with the shared library /
 * framework environment of OS X and iOS.
 *
 * Instead, if you wish to change the default BLAS error handling (which is to
 * print an english error message and exit( ););, you need to install your own
 * error handler by calling SetBLASParamErrorProc.
 *
 * Your error handler should adhere to the BLASParamErrorProc interface; it
 * need not terminate program execution.  If your error handler returns normally,
 * then the BLAS call will return normally following its execution without
 * performing any further processing.                                         */

typedef void (*BLASParamErrorProc)(const char * funcName, const char * paramName,
                                   const __LAPACK_int * paramPos,  const __LAPACK_int * paramValue);

void SetBLASParamErrorProc(BLASParamErrorProc __ErrorProc);

// MARK: BLAS Level 1

// MARK: AMAX
__LAPACK_int cblas_isamax(const __LAPACK_int N, const float * X, const __LAPACK_int INCX);
__LAPACK_int cblas_idamax(const __LAPACK_int N, const double * X, const __LAPACK_int INCX);
__LAPACK_int cblas_icamax(const __LAPACK_int N, void * X, const __LAPACK_int INCX);
__LAPACK_int cblas_izamax(const __LAPACK_int N, void * X, const __LAPACK_int INCX);

// MARK: ASUM
float cblas_sasum(const __LAPACK_int N, const float * X, const __LAPACK_int INCX);
double cblas_dasum(const __LAPACK_int N, const double * X, const __LAPACK_int INCX);
float  cblas_scasum(const __LAPACK_int N, void * X, const __LAPACK_int INCX);
double cblas_dzasum(const __LAPACK_int N, void * X, const __LAPACK_int INCX);


// MARK: AXPY
void cblas_saxpy(const __LAPACK_int N, const float ALPHA, const float * X,
                 const __LAPACK_int INCX, float * Y, const __LAPACK_int INCY);
void cblas_daxpy(const __LAPACK_int N, const double ALPHA, const double * X,
                 const __LAPACK_int INCX, double * Y, const __LAPACK_int INCY);
void cblas_caxpy(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX, void * Y, const __LAPACK_int INCY);
void cblas_zaxpy(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX, void * Y, const __LAPACK_int INCY);


// MARK: AXPBY
void catlas_saxpby(const __LAPACK_int N, const float ALPHA, const float * X,
                   const __LAPACK_int INCX, const float BETA, float * Y, const __LAPACK_int INCY);
void catlas_daxpby(const __LAPACK_int N, const double ALPHA, const double * X,
                   const __LAPACK_int INCX, const double BETA, double * Y, const __LAPACK_int INCY);
void catlas_caxpby(const __LAPACK_int N, void * ALPHA, void * X,
                   const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);
void catlas_zaxpby(const __LAPACK_int N, void * ALPHA, void * X,
                   const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);

// MARK: COPY
void cblas_scopy(const __LAPACK_int N, const float * X, const __LAPACK_int INCX,
                 float * Y, const __LAPACK_int INCY);
void cblas_dcopy(const __LAPACK_int N, const double * X, const __LAPACK_int INCX,
                 double * Y, const __LAPACK_int INCY);
void cblas_ccopy(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY);
void cblas_zcopy(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY);

// MARK: DOT
float cblas_sdot(const __LAPACK_int N, const float * X, const __LAPACK_int INCX,
                 const float * Y, const __LAPACK_int INCY);
double cblas_ddot(const __LAPACK_int N, const double * X, const __LAPACK_int INCX,
                  const double * Y, const __LAPACK_int INCY);

// MARK: DOTU
void cblas_cdotu_sub(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                     void * Y, const __LAPACK_int INCY, void * DOTU);
void cblas_zdotu_sub(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                     void * Y, const __LAPACK_int INCY, void * DOTU);

// MARK: DOTC
void cblas_cdotc_sub(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                     void * Y, const __LAPACK_int INCY, void * DOTC);
void cblas_zdotc_sub(const __LAPACK_int N, void * X, const __LAPACK_int INCX,
                     void * Y, const __LAPACK_int INCY, void * DOTC);

// MARK: DSDOT
float cblas_sdsdot(const __LAPACK_int N, const float ALPHA, const float * X,
                   const __LAPACK_int INCX, const float * Y, const __LAPACK_int INCY);

double cblas_dsdot(const __LAPACK_int N, const float * X, const __LAPACK_int INCX,
                   const float * Y, const __LAPACK_int INCY);

// MARK: NRM2
float cblas_snrm2(const __LAPACK_int N, const float * X, const __LAPACK_int INCX);
double cblas_dnrm2(const __LAPACK_int N, const double * X, const __LAPACK_int INCX);
float  cblas_scnrm2(const __LAPACK_int N, void * X, const __LAPACK_int INCX);
double cblas_dznrm2(const __LAPACK_int N, void * X, const __LAPACK_int INCX);

// MARK: ROT
void cblas_srot(const __LAPACK_int N, float * X, const __LAPACK_int INCX, float * Y,
                const __LAPACK_int INCY, const float C, const float S);
void cblas_drot(const __LAPACK_int N, double * X, const __LAPACK_int INCX, double * Y,
                const __LAPACK_int INCY, const double C, const double S);
void cblas_csrot(const __LAPACK_int N, void * X, const __LAPACK_int INCX, void * Y,
                 const __LAPACK_int INCY, const float C, const float S);
void cblas_zdrot(const __LAPACK_int N, void * X, const __LAPACK_int INCX, void * Y,
                 const __LAPACK_int INCY, const double C, const double S);

// MARK: ROTG
void cblas_srotg(float * A, float * B, float * C, float * S);
void cblas_drotg(double * A, double * B, double * C, double * S);
void cblas_crotg(void * A, void * B, float * C, void * S);
void cblas_zrotg(void * A, void * B, double * C, void * S);

// MARK: ROTM
void cblas_srotm(const __LAPACK_int N, float * X, const __LAPACK_int INCX, float * Y,
                 const __LAPACK_int INCY, const float * P);
void cblas_drotm(const __LAPACK_int N, double * X, const __LAPACK_int INCX, double * Y,
                 const __LAPACK_int INCY, const double * P);

// MARK: ROTMG
void cblas_srotmg(float * D1, float * D2, float * B1, const float B2,
                  float * P);
void cblas_drotmg(double * D1, double * D2, double * B1, const double B2,
                  double * P);

// MARK: SCAL
void cblas_sscal(const __LAPACK_int N, const float ALPHA, float * X,
                 const __LAPACK_int INCX);
void cblas_dscal(const __LAPACK_int N, const double ALPHA, double * X,
                 const __LAPACK_int INCX);
void cblas_cscal(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX);
void cblas_zscal(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX);
void cblas_csscal(const __LAPACK_int N, const float ALPHA, void * X,
                  const __LAPACK_int INCX);
void cblas_zdscal(const __LAPACK_int N, const double ALPHA, void * X,
                  const __LAPACK_int INCX);

// MARK: SET
void catlas_sset(const __LAPACK_int N, const float ALPHA, float * X,
                 const __LAPACK_int INCX);
void catlas_dset(const __LAPACK_int N, const double ALPHA, double * X,
                 const __LAPACK_int INCX);
void catlas_cset(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX);
void catlas_zset(const __LAPACK_int N, void * ALPHA, void * X,
                 const __LAPACK_int INCX);

// MARK: SWAP
void cblas_sswap(const __LAPACK_int N, float * X, const __LAPACK_int INCX, float * Y,
                 const __LAPACK_int INCY);
void cblas_dswap(const __LAPACK_int N, double * X, const __LAPACK_int INCX, double * Y,
                 const __LAPACK_int INCY);
void cblas_cswap(const __LAPACK_int N, void * X, const __LAPACK_int INCX, void * Y,
                 const __LAPACK_int INCY);
void cblas_zswap(const __LAPACK_int N, void * X, const __LAPACK_int INCX, void * Y,
                 const __LAPACK_int INCY);


// MARK: BLAS Level 2


// MARK: GEMV
void cblas_sgemv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const float ALPHA, const float * A, const __LAPACK_int LDA,
                 const float * X, const __LAPACK_int INCX, const float BETA, float * Y,
                 const __LAPACK_int INCY);
void cblas_dgemv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const double ALPHA, const double * A, const __LAPACK_int LDA,
                 const double * X, const __LAPACK_int INCX, const double BETA, double * Y,
                 const __LAPACK_int INCY);
void cblas_cgemv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);
void cblas_zgemv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);

// MARK: GBMV
void cblas_sgbmv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int KL, const __LAPACK_int KU, const float ALPHA, const float * A,
                 const __LAPACK_int LDA, const float * X, const __LAPACK_int INCX,
                 const float BETA, float * Y, const __LAPACK_int INCY);
void cblas_dgbmv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int KL, const __LAPACK_int KU, const double ALPHA,
                 const double * A, const __LAPACK_int LDA, const double * X,
                 const __LAPACK_int INCX, const double BETA, double * Y, const __LAPACK_int INCY);
void cblas_cgbmv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int KL, const __LAPACK_int KU, void * ALPHA, void * A,
                 const __LAPACK_int LDA, void * X, const __LAPACK_int INCX, void * BETA,
                 void * Y, const __LAPACK_int INCY);
void cblas_zgbmv(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int KL, const __LAPACK_int KU, void * ALPHA, void * A,
                 const __LAPACK_int LDA, void * X, const __LAPACK_int INCX, void * BETA,
                 void * Y, const __LAPACK_int INCY);
// MARK: GER
void cblas_sger(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                const float ALPHA, const float * X, const __LAPACK_int INCX,
                const float * Y, const __LAPACK_int INCY, float * A, const __LAPACK_int LDA);
void cblas_dger(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                const double ALPHA, const double * X, const __LAPACK_int INCX,
                const double * Y, const __LAPACK_int INCY, double * A, const __LAPACK_int LDA);
// MARK: GERC
void cblas_cgerc(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
void cblas_zgerc(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
// MARK: GERU
void cblas_cgeru(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
void cblas_zgeru(const enum CBLAS_ORDER ORDER, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
// MARK: HBMV
void cblas_chbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const __LAPACK_int K, void * ALPHA, void * A,
                 const __LAPACK_int LDA, void * X, const __LAPACK_int INCX, void * BETA,
                 void * Y, const __LAPACK_int INCY);
void cblas_zhbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const __LAPACK_int K, void * ALPHA, void * A,
                 const __LAPACK_int LDA, void * X, const __LAPACK_int INCX, void * BETA,
                 void * Y, const __LAPACK_int INCY);
// MARK: HEMV
void cblas_chemv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX, void * BETA, void * Y,
                 const __LAPACK_int INCY);
void cblas_zhemv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX, void * BETA, void * Y,
                 const __LAPACK_int INCY);
// MARK: HER
void cblas_cher(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const float ALPHA, void * X, const __LAPACK_int INCX,
                void * A, const __LAPACK_int LDA);
void cblas_zher(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const double ALPHA, void * X, const __LAPACK_int INCX,
                void * A, const __LAPACK_int LDA);
// MARK: HER2
void cblas_cher2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
void cblas_zher2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * A, const __LAPACK_int LDA);
// MARK: HPMV
void cblas_chpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * AP, void * X,
                 const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);
void cblas_zhpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * AP, void * X,
                 const __LAPACK_int INCX, void * BETA, void * Y, const __LAPACK_int INCY);
// MARK: HPR
void cblas_chpr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const float ALPHA, void * X, const __LAPACK_int INCX,
                void * A);
void cblas_zhpr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const double ALPHA, void * X, const __LAPACK_int INCX,
                void * A);
// MARK: HPR2
void cblas_chpr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * AP);
void cblas_zhpr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, void * ALPHA, void * X, const __LAPACK_int INCX,
                 void * Y, const __LAPACK_int INCY, void * AP);
// MARK: SBMV
void cblas_ssbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const __LAPACK_int K, const float ALPHA, const float * A,
                 const __LAPACK_int LDA, const float * X, const __LAPACK_int INCX,
                 const float BETA, float * Y, const __LAPACK_int INCY);
void cblas_dsbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const __LAPACK_int K, const double ALPHA, const double * A,
                 const __LAPACK_int LDA, const double * X, const __LAPACK_int INCX,
                 const double BETA, double * Y, const __LAPACK_int INCY);
// MARK: SPMV
void cblas_sspmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const float ALPHA, const float * AP,
                 const float * X, const __LAPACK_int INCX, const float BETA, float * Y,
                 const __LAPACK_int INCY);
void cblas_dspmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const double ALPHA, const double * AP,
                 const double * X, const __LAPACK_int INCX, const double BETA, double * Y,
                 const __LAPACK_int INCY);
// MARK: SPR
void cblas_sspr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const float ALPHA, const float * X, const __LAPACK_int INCX,
                float * AP);
void cblas_dspr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const double ALPHA, const double * X,
                const __LAPACK_int INCX, double * AP);
// MARK: SPR2
void cblas_sspr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const float ALPHA, const float * X, const __LAPACK_int INCX,
                 const float * Y, const __LAPACK_int INCY, float * A);
void cblas_dspr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const double ALPHA, const double * X,
                 const __LAPACK_int INCX, const double * Y, const __LAPACK_int INCY, double * A);
// MARK: SYMV
void cblas_ssymv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const float ALPHA, const float * A, const __LAPACK_int LDA,
                 const float * X, const __LAPACK_int INCX, const float BETA, float * Y,
                 const __LAPACK_int INCY);
void cblas_dsymv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const double ALPHA, const double * A,
                 const __LAPACK_int LDA, const double * X, const __LAPACK_int INCX,
                 const double BETA, double * Y, const __LAPACK_int INCY);
// MARK: SYR
void cblas_ssyr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const float ALPHA, const float * X, const __LAPACK_int INCX,
                float * A, const __LAPACK_int LDA);
void cblas_dsyr(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                const __LAPACK_int N, const double ALPHA, const double * X,
                const __LAPACK_int INCX, double * A, const __LAPACK_int LDA);
// MARK: SYR2
void cblas_ssyr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const float ALPHA, const float * X, const __LAPACK_int INCX,
                 const float * Y, const __LAPACK_int INCY, float * A, const __LAPACK_int LDA);
void cblas_dsyr2(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const __LAPACK_int N, const double ALPHA, const double * X,
                 const __LAPACK_int INCX, const double * Y, const __LAPACK_int INCY, double * A,
                 const __LAPACK_int LDA);
// MARK: TBMV
void cblas_stbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, const float * A, const __LAPACK_int LDA,
                 float * X, const __LAPACK_int INCX);
void cblas_dtbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, const double * A, const __LAPACK_int LDA,
                 double * X, const __LAPACK_int INCX);
void cblas_ctbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX);
void cblas_ztbmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX);
// MARK: TBSV
void cblas_stbsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, const float * A, const __LAPACK_int LDA,
                 float * X, const __LAPACK_int INCX);
void cblas_dtbsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, const double * A, const __LAPACK_int LDA,
                 double * X, const __LAPACK_int INCX);
void cblas_ctbsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX);
void cblas_ztbsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const __LAPACK_int K, void * A, const __LAPACK_int LDA,
                 void * X, const __LAPACK_int INCX);
// MARK: TPMV
void cblas_stpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const float * AP, float * X, const __LAPACK_int INCX);
void cblas_dtpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const double * AP, double * X, const __LAPACK_int INCX);
void cblas_ctpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * AP, void * X, const __LAPACK_int INCX);
void cblas_ztpmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * AP, void * X, const __LAPACK_int INCX);
// MARK: TPSV
void cblas_stpsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const float * AP, float * X, const __LAPACK_int INCX);
void cblas_dtpsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const double * AP, double * X, const __LAPACK_int INCX);
void cblas_ctpsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * AP, void * X, const __LAPACK_int INCX);
void cblas_ztpsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * AP, void * X, const __LAPACK_int INCX);

// MARK: TRMV
void cblas_strmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const float * A, const __LAPACK_int LDA, float * X,
                 const __LAPACK_int INCX);
void cblas_dtrmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const double * A, const __LAPACK_int LDA, double * X,
                 const __LAPACK_int INCX);
void cblas_ctrmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX);
void cblas_ztrmv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX);
// MARK: TRSV
void cblas_strsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const float * A, const __LAPACK_int LDA, float * X,
                 const __LAPACK_int INCX);
void cblas_dtrsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, const double * A, const __LAPACK_int LDA, double * X,
                 const __LAPACK_int INCX);
void cblas_ctrsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX);
void cblas_ztrsv(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANSA, const enum CBLAS_DIAG DIAG,
                 const __LAPACK_int N, void * A, const __LAPACK_int LDA, void * X,
                 const __LAPACK_int INCX);
// MARK: BLAS Level 3

// MARK: GEMM
void cblas_sgemm(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int K, const float ALPHA, const float * A, const __LAPACK_int LDA,
                 const float * B, const __LAPACK_int LDB, const float BETA, float * C,
                 const __LAPACK_int LDC);
void cblas_dgemm(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int K, const double ALPHA, const double * A,
                 const __LAPACK_int LDA, const double * B, const __LAPACK_int LDB,
                 const double BETA, double * C, const __LAPACK_int LDC);
void cblas_cgemm(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int K, void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * B, const __LAPACK_int LDB, void * BETA, void * C,
                 const __LAPACK_int LDC);
void cblas_zgemm(const enum CBLAS_ORDER ORDER,
                 const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_TRANSPOSE TRANSB, const __LAPACK_int M, const __LAPACK_int N,
                 const __LAPACK_int K, void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * B, const __LAPACK_int LDB, void * BETA, void * C,
                 const __LAPACK_int LDC);
// MARK: HEMM
void cblas_chemm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
void cblas_zhemm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
// MARK: HERK
void cblas_cherk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 const float ALPHA, void * A, const __LAPACK_int LDA,
                 const float BETA, void * C, const __LAPACK_int LDC);
void cblas_zherk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 const double ALPHA, void * A, const __LAPACK_int LDA,
                 const double BETA, void * C, const __LAPACK_int LDC);
// MARK: HER2K
void cblas_cher2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                  const __LAPACK_int LDB, const float BETA, void * C, const __LAPACK_int LDC);
void cblas_zher2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                  const __LAPACK_int LDB, const double BETA, void * C, const __LAPACK_int LDC);
// MARK: SYMM
void cblas_ssymm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 const float ALPHA, const float * A, const __LAPACK_int LDA,
                 const float * B, const __LAPACK_int LDB, const float BETA, float * C,
                 const __LAPACK_int LDC);
void cblas_dsymm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 const double ALPHA, const double * A, const __LAPACK_int LDA,
                 const double * B, const __LAPACK_int LDB, const double BETA, double * C,
                 const __LAPACK_int LDC);
void cblas_csymm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
void cblas_zsymm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
// MARK: SYRK
void cblas_ssyrk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 const float ALPHA, const float * A, const __LAPACK_int LDA,
                 const float BETA, float * C, const __LAPACK_int LDC);
void cblas_dsyrk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 const double ALPHA, const double * A, const __LAPACK_int LDA,
                 const double BETA, double * C, const __LAPACK_int LDC);
void cblas_csyrk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * BETA, void * C, const __LAPACK_int LDC);
void cblas_zsyrk(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                 const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                 void * ALPHA, void * A, const __LAPACK_int LDA,
                 void * BETA, void * C, const __LAPACK_int LDC);
// MARK: SYR2K
void cblas_ssyr2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  const float ALPHA, const float * A, const __LAPACK_int LDA,
                  const float * B, const __LAPACK_int LDB, const float BETA, float * C,
                  const __LAPACK_int LDC);
void cblas_dsyr2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  const double ALPHA, const double * A, const __LAPACK_int LDA,
                  const double * B, const __LAPACK_int LDB, const double BETA, double * C,
                  const __LAPACK_int LDC);
void cblas_csyr2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                  const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
void cblas_zsyr2k(const enum CBLAS_ORDER ORDER, const enum CBLAS_UPLO UPLO,
                  const enum CBLAS_TRANSPOSE TRANS, const __LAPACK_int N, const __LAPACK_int K,
                  void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                  const __LAPACK_int LDB, void * BETA, void * C, const __LAPACK_int LDC);
// MARK: TRMM
void cblas_strmm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 const float ALPHA, const float * A, const __LAPACK_int LDA, float * B,
                 const __LAPACK_int LDB);
void cblas_dtrmm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 const double ALPHA, const double * A, const __LAPACK_int LDA, double * B,
                 const __LAPACK_int LDB);
void cblas_ctrmm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB);
void cblas_ztrmm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB);
// MARK: TRSM
void cblas_strsm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 const float ALPHA, const float * A, const __LAPACK_int LDA, float * B,
                 const __LAPACK_int LDB);
void cblas_dtrsm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 const double ALPHA, const double * A, const __LAPACK_int LDA, double * B,
                 const __LAPACK_int LDB);
void cblas_ctrsm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB);
void cblas_ztrsm(const enum CBLAS_ORDER ORDER, const enum CBLAS_SIDE SIDE,
                 const enum CBLAS_UPLO UPLO, const enum CBLAS_TRANSPOSE TRANSA,
                 const enum CBLAS_DIAG DIAG, const __LAPACK_int M, const __LAPACK_int N,
                 void * ALPHA, void * A, const __LAPACK_int LDA, void * B,
                 const __LAPACK_int LDB);
