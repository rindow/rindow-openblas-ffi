<?php
namespace Rindow\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use InvalidArgumentException;
use RuntimeException;
use FFI;
use ArrayObject;

use Interop\Polite\Math\Matrix\LinearBuffer as BufferInterface;

class ffi_char_t2
{
    public string $cdata;
}

class Lapackb implements Lapack
{
    use Utils;

    const LAPACK_WORK_MEMORY_ERROR      = -1010;
    const LAPACK_TRANSPOSE_MEMORY_ERROR = -1010;
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected FFI $ffi;
    protected FFI $blas;

    public function __construct(FFI $ffi, FFI $blas)
    {
        $this->ffi = $ffi;
        $this->blas = $blas;
    }

    public function ffi() : object
    {
        return $this->ffi;
    }

    /**
     * Get a temporary identity matrix (ColMajor layout).
     * Consider caching these if memory/performance becomes an issue.
     */
    private function getIdentity(int $size, int $dtype, string $type) : FFI\CData
    {
        $identity_p = $this->ffi->new("{$type}[$size * $size]");
        // Initialize with zeros
        $this->ffi::memset($identity_p, 0, $this->ffi::sizeof($identity_p));
        // Set diagonal elements to 1.0 (ColMajor: I[i,i] is at offset i*size + i)
        for($i = 0; $i < $size; $i++) {
            $identity_p[$i * $size + $i] = 1.0;
        }
        return $identity_p;
    }
    
    /**
     * Transposes a matrix using BLAS GEMM.
     * Copies from RowMajor (m x n, ldA=n) to ColMajor (m x n, ldB=m).
     * B_col = A_row^T * I_n (interpreted in ColMajor)
     */
    private function transpose_row_to_col_gemm(
        int $m, int $n, int $dtype,
        FFI\CData $A_ptr, int $ldA, // Source RowMajor data pointer and ld (n)
        FFI\CData $B_ptr, int $ldB  // Destination ColMajor data pointer and ld (m)
    ) : void
    {
        if ($ldA < $n) throw new InvalidArgumentException("transpose_row_to_col_gemm: ldA must be >= n");
        if ($ldB < $m) throw new InvalidArgumentException("transpose_row_to_col_gemm: ldB must be >= m");
    
        $type = ($dtype == NDArray::float32) ? 'float' : 'double';
        $gemm_func = ($dtype == NDArray::float32) ? 'cblas_sgemm' : 'cblas_dgemm';
    
        $identity_n = $this->getIdentity($n, $dtype, $type);
        $ldI_n = $n; // ColMajor identity matrix leading dimension
    
        // We use CblasColMajor to avoid the ldc issue encountered with CblasRowMajor.
        // Operation: C(m,n) = A(m,n)^T * B(n,n) where A is implicitly ColMajor(n,m) ld=n.
        // Arguments for cblas_?gemm(ColMajor, TransA, NoTransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC):
        // M = m (rows of op(A) and C)
        // N = n (cols of op(B) and C)
        // K = n (cols of op(A) = rows of op(B))
        // A = A_ptr (interpreted as ColMajor n x m, ldA = n)
        // ldA = n
        // B = identity_n (ColMajor n x n, ldB = n)
        // ldB = n
        // C = B_ptr (output ColMajor m x n, ldC = m)
        // ldC = m >= M = m (Constraint is met)
        $this->blas->{$gemm_func}(
            BLAS::ColMajor, // Or use integer constant 102
            BLAS::Trans,    // Or use integer constant 112
            BLAS::NoTrans,  // Or use integer constant 111
            $m, $n, $n,
            1.0,
            $A_ptr, $ldA,       // A is RowMajor(m,n) but treated as ColMajor(n,m) ld=n
            $identity_n, $ldI_n,
            0.0,
            $B_ptr, $ldB        // C is ColMajor(m,n) ld=m
        );
    }
    
    /**
     * Transposes a matrix using BLAS GEMM.
     * Copies from ColMajor (m x n, ldA=m) to RowMajor (m x n, ldB=n).
     * B_row = A_col^T * I_m (interpreted in ColMajor)
     */
    private function transpose_col_to_row_gemm(
        int $m, int $n, int $dtype,
        FFI\CData $A_ptr, int $ldA, // Source ColMajor data pointer and ld (m)
        FFI\CData $B_ptr, int $ldB  // Destination RowMajor data pointer and ld (n)
    ) : void
    {
        if ($ldA < $m) throw new InvalidArgumentException("transpose_col_to_row_gemm: ldA must be >= m");
        if ($ldB < $n) throw new InvalidArgumentException("transpose_col_to_row_gemm: ldB must be >= n");
    
        $type = ($dtype == NDArray::float32) ? 'float' : 'double';
        $gemm_func = ($dtype == NDArray::float32) ? 'cblas_sgemm' : 'cblas_dgemm';
    
        $identity_m = $this->getIdentity($m, $dtype, $type);
        $ldI_m = $m; // ColMajor identity matrix leading dimension
    
        // We use CblasColMajor.
        // Operation: C(n,m) = A(m,n)^T * B(m,m) where A is ColMajor(m,n) ld=m.
        // The output C(n,m) in ColMajor (ld=n) has the same memory layout as the desired RowMajor B(m,n) ld=n.
        // Arguments for cblas_?gemm(ColMajor, TransA, NoTransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC):
        // M = n (rows of op(A) and C)
        // N = m (cols of op(B) and C)
        // K = m (cols of op(A) = rows of op(B))
        // A = A_ptr (ColMajor m x n, ldA = m)
        // ldA = m
        // B = identity_m (ColMajor m x m, ldB = m)
        // ldB = m
        // C = B_ptr (output ColMajor n x m, ldC = n) -> Effectively RowMajor(m,n) ld=n
        // ldC = n >= M = n (Constraint is met)
         $this->blas->{$gemm_func}(
            BLAS::ColMajor, // Or use integer constant 102
            BLAS::Trans,    // Or use integer constant 112
            BLAS::NoTrans,  // Or use integer constant 111
            $n, $m, $m,
            1.0,
            $A_ptr, $ldA,       // A is ColMajor(m,n) ld=m
            $identity_m, $ldI_m,
            0.0,
            $B_ptr, $ldB        // C is ColMajor(n,m) ld=n (target buffer for RowMajor m x n)
        );
    }

    public function gesvd(
        int $matrix_layout,
        int $jobu, // ord('A') or ord('S')
        int $jobvt, // ord('A') or ord('S')
        int $m,
        int $n,
        BufferInterface $A,  int $offsetA,  int $ldA, // For ROW_MAJOR, ldA=$n; For COL_MAJOR, ldA=$m
        BufferInterface $S,  int $offsetS,
        BufferInterface $U,  int $offsetU,  int $ldU, // For ROW_MAJOR, ldU=colsU; For COL_MAJOR, ldU=$m
        BufferInterface $VT, int $offsetVT, int $ldVT, // For ROW_MAJOR, ldVT=$n; For COL_MAJOR, ldVT=rowsVT
        BufferInterface $SuperB,  int $offsetSuperB
    ) : void
    {
        $ffi = $this->ffi;
        // ... (Initial checks for shape, offsets, buffers, dtype) ...
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        if( $offsetS < 0 ) {
            throw new InvalidArgumentException("offsetS must be greater than zero or equal");
        }
        if( $offsetU < 0 ) {
            throw new InvalidArgumentException("offsetU must be greater than zero or equal", 0);
        }
        if( $ldU <= 0 ) {
            throw new InvalidArgumentException("ldU must be greater than zero", 0);
        }
        if( $offsetVT < 0 ) {
            throw new InvalidArgumentException("offsetVT must be greater than zero or equal", 0);
        }
        if( $ldVT <= 0 ) {
            throw new InvalidArgumentException("ldVT must be greater than zero", 0);
        }
        if( $offsetSuperB < 0 ) {
            throw new InvalidArgumentException("offsetVT must be greater than zero or equal", 0);
        }
        // Check Buffer A
        $this->assert_matrix_buffer_spec("BufferA", $A,$m,$n,$offsetA,$ldA);
    
        // Check Buffer S
        if( $offsetS+min($m,$n) > count($S)) {
            throw new InvalidArgumentException("BufferS size is too small", 0);
        }
    
        // Check Buffer U
        if( $offsetU+$m*$ldU > count($U)) {
            throw new InvalidArgumentException("BufferU size is too small", 0);
        }
    
        // Check Buffer VT
        if( $offsetVT+$ldVT*$n > count($VT)) {
            throw new InvalidArgumentException("BufferVT size is too small", 0);
        }
    
        // Check Buffer SuperB
        if( $offsetSuperB+min($m,$n)-1 > count($SuperB)) {
            throw new InvalidArgumentException("bufferSuperB size is too small", 0);
        }

        $k = min($m, $n);
        $dtype = $A->dtype();
        if($dtype==NDArray::float32) {
            $type = 'float';
            $gesvd_func = 'sgesvd_';
        } elseif($dtype==NDArray::float64) {
            $type = 'double';
            $gesvd_func = 'dgesvd_';
        } else {
            throw new InvalidArgumentException("Unsupported data type", 0);
        }

        $targetA_ptr = null; // Pointer for COL_MAJOR A (if layout is RowMajor)
        $targetU_ptr = null; // Pointer for COL_MAJOR U (if layout is RowMajor)
        $targetVT_ptr = null;// Pointer for COL_MAJOR VT (if layout is RowMajor)
        $ptrA = null;      // Pointer to A data for gesvd_
        $ptrU = null;      // Pointer to U data for gesvd_
        $ptrVT = null;     // Pointer to VT data for gesvd_
        $ldA0 = 0;         // Leading dimension for A in gesvd_ (ColMajor ld = rows)
        $ldU0 = 0;         // Leading dimension for U in gesvd_ (ColMajor ld = rows)
        $ldVT0 = 0;        // Leading dimension for VT in gesvd_ (ColMajor ld = rows)

        // Determine dimensions for sgesvd_ output (COL_MAJOR perspective)
        // Actual number of columns computed for U (m x m or m x k)
        $colsU_computed = ($jobu == ord('A')) ? $m : $k;
        // Actual number of rows computed for VT (n x n or k x n)
        $rowsVT_computed = ($jobvt == ord('A')) ? $n : $k;

        if($matrix_layout == self::LAPACK_ROW_MAJOR) {
            // --- Input Transpose: RowMajor A -> ColMajor targetA_ptr ---
            $sizeA = $m * $n;
            $targetA_ptr = $ffi->new("{$type}[{$sizeA}]");
            $ldA_col = $m; // Target ColMajor LD is rows (m)
            $this->transpose_row_to_col_gemm($m, $n, $dtype, $A->addr($offsetA), $ldA, $targetA_ptr, $ldA_col);
            $ptrA = $targetA_ptr;
            $ldA0 = $ldA_col; // LD for gesvd_ is m

            // --- Allocate temporary ColMajor buffers for U and VT ---
            $sizeU = $m * $colsU_computed;
            $targetU_ptr = $ffi->new("{$type}[{$sizeU}]");
            $ldU0 = $m; // gesvd_ needs ColMajor LD (rows)
            $ptrU = $targetU_ptr;

            // VT is rowsVT_computed x n (ColMajor layout), ld = rowsVT_computed
            $ldVT0 = $rowsVT_computed;
            $sizeVT = $ldVT0 * $n;
            $targetVT_ptr = $ffi->new("{$type}[{$sizeVT}]");
            $ptrVT = $targetVT_ptr;

        } elseif($matrix_layout == self::LAPACK_COL_MAJOR) {
            // Data is already in COL_MAJOR, use buffers directly
            $ptrA = $A->addr($offsetA);
            $ldA0 = $ldA; // Caller provided ColMajor ld (m)
            $ptrU = $U->addr($offsetU);
            $ldU0 = $ldU; // Caller provided ColMajor ld (m)
            $ptrVT = $VT->addr($offsetVT);
            $ldVT0 = $ldVT; // Caller provided ColMajor ld (rowsVT_computed)
        } else {
            throw new InvalidArgumentException("Invalid matrix_layout: $matrix_layout");
        }

        // Prepare parameters for gesvd_
        $jobu_p = $ffi->new('char[1]'); $jobu_p[0] = chr($jobu);
        $jobvt_p = $ffi->new('char[1]'); $jobvt_p[0] = chr($jobvt);
        $m_p = $ffi->new('lapack_int[1]'); $m_p[0] = $m;
        $n_p = $ffi->new('lapack_int[1]'); $n_p[0] = $n;
        $ldA_p = $ffi->new('lapack_int[1]'); $ldA_p[0] = $ldA0;
        $ldU_p = $ffi->new('lapack_int[1]'); $ldU_p[0] = $ldU0;
        $ldVT_p = $ffi->new('lapack_int[1]'); $ldVT_p[0] = $ldVT0; // Use ColMajor ldVT0
        $info_p = $ffi->new("lapack_int[1]"); $info_p[0] = 0;
        $lwork_p = $ffi->new("lapack_int[1]"); $lwork_p[0] = -1;
        $wkopt_p = $ffi->new("{$type}[1]"); // For workspace query

        // --- Workspace query ---
        $ffi->{$gesvd_func}(
            $jobu_p, $jobvt_p, $m_p, $n_p,
            $ptrA, $ldA_p,
            $S->addr($offsetS),
            $ptrU, $ldU_p,
            $ptrVT, $ldVT_p, // Pass correct ColMajor ldVT0
            $wkopt_p, $lwork_p, $info_p
        );
        $info = $info_p[0];
        if ($info != 0) {
            throw new RuntimeException("gesvd_ workspace query failed. error=$info", $info);
        }

        $lwork = (int)$wkopt_p[0];
        $lwork_p[0] = $lwork;
        $work = $ffi->new("{$type}[{$lwork}]");
        $info_p[0] = 0; // Reset info

        // --- Actual gesvd_ call ---
        $ffi->{$gesvd_func}(
            $jobu_p, $jobvt_p, $m_p, $n_p,
            $ptrA, $ldA_p,
            $S->addr($offsetS),
            $ptrU, $ldU_p,
            $ptrVT, $ldVT_p, // Pass correct ColMajor ldVT0
            $work, $lwork_p, $info_p
        );
        $info = $info_p[0];
        // Check info for errors (negative values) or convergence issues (positive values)
        if ($info < 0) {
            throw new RuntimeException("gesvd_ parameter error. argument ".(-$info)." had an illegal value.", $info);
        }
        if ($info > 0) {
            /* Handle convergence failure if needed, e.g., log a warning */
            error_log("Warning: gesvd_ failed to converge. ".$info." superdiagonals did not converge.");
        }

        // --- SuperB copy (optional, usually internal detail) ---
        // $superb_len = $k - 1;
        // if ($superb_len > 0 && count($SuperB) >= $superb_len) { ... } // Be cautious if implementing

        // --- Output Transpose (if input was RowMajor) ---
        if($matrix_layout == self::LAPACK_ROW_MAJOR) {
            // U: targetU_ptr (ColMajor, m x colsU_computed, ldU0=m) -> U buffer (RowMajor, m x colsU_computed, ldU=colsU_computed)
            $this->transpose_col_to_row_gemm($m, $colsU_computed, $dtype, $ptrU, $ldU0, $U->addr($offsetU), $ldU);

            // VT: targetVT_ptr (ColMajor, rowsVT_computed x n, ldVT0=rowsVT_computed) -> VT buffer (RowMajor, rowsVT_computed x n, ldVT=n)
            $this->transpose_col_to_row_gemm($rowsVT_computed, $n, $dtype, $ptrVT, $ldVT0, $VT->addr($offsetVT), $ldVT);
        }
        // If layout was COL_MAJOR, results are already in the provided U, VT buffers.
        // Temporary FFI CData ($targetA_ptr, $targetU_ptr, $targetVT_ptr, $work, etc.) will be garbage collected.
    }
}
