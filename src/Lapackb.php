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

    private function transpose(
        int $m, int $n, int $dtype,
        object $A, int $ldA,
        object $B, int $ldB
        ) : void
    {
        $ffi = $this->ffi;
        $blas = $this->blas;
        if($dtype==NDArray::float32) {
            $type = 'float';
        } elseif($dtype==NDArray::float64) {
            $type = 'double';
        } else {
            throw new InvalidArgumentException("Unsupport data type", 0);
        }
        $size = $m*$m;
        $identity_p = $ffi->new("{$type}[{$size}]");
        $ffi::memset($identity_p,0,$ffi::sizeof($identity_p));
        for($i=0;$i<$m;$i++) {
            $identity_p[$i*$m+$i] = 1;
        }
        if($dtype==NDArray::float32) {
            $blas->cblas_sgemm(
                self::LAPACK_ROW_MAJOR,BLAS::Trans,BLAS::NoTrans,
                $n,$m,$m,
                1.0,
                $A,$ldA,
                $identity_p,$m,
                0.0,
                $B,$ldB
            );
        } else {
            $blas->cblas_dgemm(
                self::LAPACK_ROW_MAJOR,BLAS::Trans,BLAS::NoTrans,
                $n,$m,$m,
                1.0,
                $A,$ldA,
                $identity_p,$m,
                0.0,
                $B,$ldB
            );
        }

    }

    public function gesvd(
        int $matrix_layout,
        int $jobu,
        int $jobvt,
        int $m,
        int $n,
        BufferInterface $A,  int $offsetA,  int $ldA,
        BufferInterface $S,  int $offsetS,
        BufferInterface $U,  int $offsetU,  int $ldU,
        BufferInterface $VT, int $offsetVT, int $ldVT,
        BufferInterface $SuperB,  int $offsetSuperB
    ) : void
    {
        $ffi = $this->ffi;
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
    
        $dtype = $A->dtype();
        // Check Buffer A and B and C
        if($dtype!=$S->dtype() ||
            $dtype!=$U->dtype() ||
            $dtype!=$VT->dtype() ||
            $dtype!=$SuperB->dtype()
        ) {
            throw new InvalidArgumentException("Unmatch data type", 0);
        }
        if($dtype==NDArray::float32) {
            $type = 'float';
        } elseif($dtype==NDArray::float64) {
            $type = 'double';
        } else {
            throw new InvalidArgumentException("Unsupport data type", 0);
        }
        if($matrix_layout==self::LAPACK_ROW_MAJOR) {
            $size = $m*$n;
            $targetA_p = $ffi->new("{$type}[{$size}]");
            $this->transpose($m,$n,$dtype,$A->addr($offsetA),$ldA,$targetA_p,$m);
            $size = $m*$m;
            $targetU_p = $ffi->new("{$type}[{$size}]");
            $size = $n*$n;
            $targetVT_p = $ffi->new("{$type}[{$size}]");
            $ldA0 = $m;
            $ldU0 = $m;
            $ldVT0 = $n;
        } elseif($matrix_layout==self::LAPACK_COL_MAJOR) {
            $targetA_p = $A->addr($offsetA);
            $targetU_p = $U->addr($offsetU);
            $targetVT_p = $VT->addr($offsetVT);
            $ldA0 = $ldA;
            $ldU0 = $ldU;
            $ldVT0 = $ldVT;
        } else {
            throw new InvalidArgumentException("Invalid matrix_layout: $matrix_layout");
        }
        //echo "------ targetA -----\n";
        //echo "[\n";
        //for($i=0;$i<$m;$i++) {
        //    echo "[";
        //    for($j=0;$j<$n;$j++) {
        //        echo sprintf('%10.6f',$targetA_p[$i+$j*$m]).",";
        //    }
        //    echo "]\n";
        //}
        //echo "]\n";


        /** @var ffi_char_t2 $jobu_p */
        $jobu_p = $ffi->new('char[4]');
        $jobu_p[0] = chr($jobu);
        /** @var ffi_char_t2 $jobvt_p */
        $jobvt_p = $ffi->new('char[4]');
        $jobvt_p[0] = chr($jobvt);
        $m_p = $ffi->new('lapack_int[1]');
        $m_p[0] = $m;
        $n_p = $ffi->new('lapack_int[1]');
        $n_p[0] = $n;
        $ldA_p = $ffi->new('lapack_int[1]');
        $ldA_p[0] = $ldA0;
        $ldU_p = $ffi->new('lapack_int[1]');
        $ldU_p[0] = $ldU0;
        $ldVT_p = $ffi->new('lapack_int[1]');
        $ldVT_p[0] = $ldVT0;
        $lwork_p = $ffi->new("lapack_int[1]");
        $lwork_p[0] = -1;
        /** @var ArrayObject<int,int> $info_p */
        $info_p = $ffi->new("lapack_int[1]");
        $info_p[0] = 0;

        //var_dump($ffi::sizeof($m_p)*8);
        //return;
        switch ($dtype) {
            case NDArray::float32:
                $wkopt_p = $ffi->new("{$type}[1]");
                $ffi->sgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA_p, $ldA_p,
                    $S->addr($offsetS),
                    $targetU_p, $ldU_p,
                    $targetVT_p, $ldVT_p,
                    $wkopt_p,$lwork_p,
                    $info_p
                );
                break;
            case NDArray::float64:
                $wkopt_p = $ffi->new("{$type}[1]");
                $ffi->dgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA_p, $ldA_p,
                    $S->addr($offsetS),
                    $targetU_p, $ldU_p,
                    $targetVT_p, $ldVT_p,
                    $wkopt_p,$lwork_p,
                    $info_p
                );
                break;
            default:
                throw new RuntimeException("Unsupported data type.", 0);
        }
        $info = $info_p[0];
        if( $info == self::LAPACK_WORK_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to allocate work array.", $info);
        } else if( $info == self::LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to transpose matrix.", $info);
        } else if( $info < 0 ) {
            throw new RuntimeException( "Wrong parameter. error=$info", $info);
        }
        $info_p[0] = 0;
        $lwork = (int)$wkopt_p[0];
        $lwork_p[0]= $lwork;

        switch ($dtype) {
            case NDArray::float32:
                $work = $ffi->new("{$type}[{$lwork}]");
                $ffi->sgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA_p, $ldA_p,
                    $S->addr($offsetS),
                    $targetU_p, $ldU_p,
                    $targetVT_p, $ldVT_p,
                    $work,$lwork_p,
                    $info_p
                );
                break;
            case NDArray::float64:
                $work = $ffi->new("{$type}[{$lwork}]");
                $ffi->dgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA_p, $ldA_p,
                    $S->addr($offsetS),
                    $targetU_p, $ldU_p,
                    $targetVT_p, $ldVT_p,
                    $work,$lwork_p,
                    $info_p
                );
                break;
            default:
                throw new RuntimeException("Unsupported data type.", 0);
        }
        $info = $info_p[0];

        if( $info == self::LAPACK_WORK_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to allocate work array.", $info);
        } else if( $info == self::LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to transpose matrix.", $info);
        } else if( $info != 0 ) {
            throw new RuntimeException( "Wrong parameter. error=$info", $info);
        }

        $bytes = min(min($m,$n)-1,$lwork-1)*$ffi::sizeof($ffi->type("{$type}"));
        $ffi::memcpy($SuperB->addr($offsetSuperB),$ffi::addr($work[1]),$bytes);
        if($matrix_layout==self::LAPACK_ROW_MAJOR) {
            $this->transpose($m,$m,$dtype,$targetU_p,$m,$U->addr($offsetU),$ldU);
            $this->transpose($n,$n,$dtype,$targetVT_p,$n,$VT->addr($offsetVT),$ldVT);
        }
    }
}
