<?php
namespace RindowTest\OpenBLAS\FFI\BlasTest;

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\RequiresOperatingSystem;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\OpenBLAS\FFI\Blas as OpenBLAS;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use InvalidArgumentException;
use TypeError;
use FFI;

require_once __DIR__.'/Utils.php';
use RindowTest\OpenBLAS\FFI\Utils;
use function RindowTest\OpenBLAS\FFI\C;

class BlasTest extends TestCase
{
    use Utils;

    public function getOpenBLASVersion($blas)
    {
        $config = $blas->getConfig();
        if(strpos($config,'OpenBLAS')===0) {
            $config = explode(' ',$config);
            return $config[1];
        } else {
            return '0.0.0';
        }
    }

    protected function skipComplex() : bool
    {
        return false;
        #if(PHP_OS!=='Darwin') {
        #    return false;
        #}
        #$this->markTestSkipped("skip Complex");
        #return true;
    }

    protected function notSupportComplex() : bool
    {
        return false;
        #if(PHP_OS!=='Darwin') {
        #    return false;
        #}
        #return true;
    }

    public function skipiamin()
    {
        $blas = $this->getBlas(null);
        if(version_compare($this->getOpenBLASVersion($blas),'0.3.6','>=')) {
            return false;
        }
        $this->markTestSkipped("openblas has no iamin");
        return true;
    }

    public function translate_dot(
        NDArray $X,NDArray $Y) : array
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $N = $X->size();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        return [$N,$XX,$offX,1,$YY,$offY,1];
    }

    public function translate_rotg(
        NDArray $X,
        NDArray $Y,
        ?NDArray $R=null,
        ?NDArray $Z=null,
        ?NDArray $C=null,
        ?NDArray $S=null) : array
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $R = $this->copy($X,$R);
        $Z = $this->copy($Y,$Z);
        if($C==null) {
            $C = $this->alloc($X->shape(),$X->dtype());
        }
        if($S==null) {
            $S = $this->alloc($Y->shape(),$X->dtype());
        }
        $AA = $R->buffer();
        $offA = $R->offset();
        $BB = $Z->buffer();
        $offB = $Z->offset();
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();
        return [
            $AA,$offA,
            $BB,$offB,
            $CC,$offC,
            $SS,$offS
        ];
    }

    public function translate_rot(
        NDArray $X,
        NDArray $Y,
        NDArray $C,
        NDArray $S) : array
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $CC = $C->buffer();
        $offC = $C->offset();
        $SS = $S->buffer();
        $offS = $S->offset();

        return [ $N,
            $XX,$offX,1,$YY,$offY,1,
            $CC,$offC,$SS,$offS
        ];
    }

    public function translate_rotmg(
        NDArray $X,
        NDArray $Y,
        ?NDArray $D1=null,
        ?NDArray $D2=null,
        ?NDArray $B1=null,
        ?NDArray $P=null,
        ) : array
    {
        if($X->size()!=1||$Y->size()!=1) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($D1==null) {
            $D1 = $this->ones([],dtype:$X->dtype());
        }
        if($D2==null) {
            $D2 = $this->ones([],dtype:$X->dtype());
        }
        if($B1==null) {
            $B1 = $this->zeros([],dtype:$X->dtype());
        }
        if($P==null) {
            $P = $this->zeros([5],dtype:$X->dtype());
        }
        $this->copy($X->reshape([1]),$B1->reshape([1]));

        $DD1 = $D1->buffer();
        $offD1 = $D1->offset();
        $DD2 = $D2->buffer();
        $offD2 = $D2->offset();
        $BB1 = $B1->buffer();
        $offB1 = $B1->offset();
        $BB1 = $B1->buffer();
        $offB1 = $B1->offset();
        $BB2 = $Y->buffer();
        $offB2 = $Y->offset();
        $PP = $P->buffer();
        $offP = $P->offset();
        return [
            $DD1,$offD1,
            $DD2,$offD2,
            $BB1,$offB1,
            $BB2,$offB2,
            $PP,$offP,
        ];
    }

    public function translate_gemv(
        NDArray $A,
        NDArray $X,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $Y=null,
        ?bool $trans=null,
        ?bool $conj=null)
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $X->ndim()!=1) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray and "X" must 1D-NDArray.');
        }
        $shapeA = $A->shape();
        $shapeX = $X->shape();
        $rows = (!$trans) ? $shapeA[0] : $shapeA[1];
        $cols = (!$trans) ? $shapeA[1] : $shapeA[0];
        if($cols!=$shapeX[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and The number of item in "X" must be the same');
        }
        $AA = $A->buffer();
        $XX = $X->buffer();
        $offA = $A->offset();
        $offX = $X->offset();
        $m = $shapeA[0];
        $n = $shapeA[1];
        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        if($beta===null) {
            if($this->isComplex($A->dtype())) {
                $beta = C(0.0);
            } else {
                $beta = 0.0;
            }
        }
        if($Y!=null) {
            if($Y->ndim()!=1) {
                throw new InvalidArgumentException('"Y" must 1D-NDArray.');
            }
            $shapeY = $Y->shape();
            if($rows!=$shapeY[0]) {
                throw new InvalidArgumentException('The number of rows in "A" and The number of item in "Y" must be the same');
            }
        } else {
            $Y = $this->mo->zeros([$rows]);
        }
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        return [
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$n,
            $XX,$offX,1,
            $beta,
            $YY,$offY,1,
        ];
    }

    public function translate_gemm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $transA=null,
        ?bool $transB=null,
        ?bool $conjA=null,
        ?bool $conjB=null,)
    {
        [$transA,$conjA] = $this->complementTrans($transA,$conjA,$A->dtype());
        [$transB,$conjB] = $this->complementTrans($transB,$conjB,$B->dtype());

        $shapeA = $A->shape();
        if($transA) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $shapeB = $B->shape();
        if($transB) {
            $shapeB = [$shapeB[1],$shapeB[0]];
        }
        if($shapeA[1]!=$shapeB[0]) {
            throw new InvalidArgumentException('The number of columns in "A" and the number of rows in "B" must be the same');
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();
        $M = $shapeA[0];
        $N = $shapeB[1];
        $K = $shapeA[1];

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        if($beta===null) {
            if($this->isComplex($A->dtype())) {
                $beta = C(0.0);
            } else {
                $beta = 0.0;
            }
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"A" and "C" must have the same number of rows."B" and "C" must have the same number of columns');
            }
        } else {
            $C = $this->mo->zeros([$M,$N]);
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($transA) ? $M : $K;
        $ldb = ($transB) ? $K : $N;
        $ldc = $N;
        $transA = $this->transToCode($transA,$conjA);
        $transB = $this->transToCode($transB,$conjB);
        $order = BLAS::RowMajor;

        return [
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc,
        ];
    }

    public function translate_symm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $right=null,
        ?bool $lower=null
        ) : array
    {
        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $rowsA = $shapeA[0];
        if($rowsA!=$shapeA[1]) {
            throw new InvalidArgumentException('The matrix "A" must be symmetric');
        }
        $shapeB = $B->shape();
        $M = $shapeB[0];
        $N = $shapeB[1];
        $tmpB = ($right) ? $N : $M;
        if($rowsA!=$tmpB) {
            throw new InvalidArgumentException('Unmatch Shape of matrix "A" and "B": '."($rowsA,$rowsA) != ($M,$N)");
        }
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offA = $A->offset();
        $offB = $B->offset();

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        if($beta===null) {
            if($this->isComplex($A->dtype())) {
                $beta = C(0.0);
            } else {
                $beta = 0.0;
            }
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($M!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('Matrix "B" and "C" must be same shape');
            }
        } else {
            $C = $this->zeros($this->alloc([$M,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = $rowsA;
        $ldb = $N;
        $ldc = $N;
        $side = ($right) ? BLAS::Right : BLAS::Left;
        $uplo = ($lower) ? BLAS::Lower : BLAS::Upper;
        $order = BLAS::RowMajor;

        return [
            $order,$side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ];
    }

    public function translate_syrk(
        NDArray $A,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ) : array
    {
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($A->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }

        $shapeA = $A->shape();
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        if($beta===null) {
            if($this->isComplex($A->dtype())) {
                $beta = C(0.0);
            } else {
                $beta = 0.0;
            }
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        return [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        ];
    }

    public function translate_syr2k(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        float|object|null $beta=null,
        ?NDArray $C=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ) : array
    {
        $trans = $trans ?? false;
        // $conj = $conj ?? $trans; // Doing so will result in an error.
        $conj = false;  // conj must be false

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($shapeA!=$shapeB) {
            throw new InvalidArgumentException('Matrix A and B must be same shape');
        }
        if($trans) {
            $shapeA = [$shapeA[1],$shapeA[0]];
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $N = $shapeA[0];
        $K = $shapeA[1];

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        if($beta===null) {
            if($this->isComplex($A->dtype())) {
                $beta = C(0.0);
            } else {
                $beta = 0.0;
            }
        }
        if($C!=null) {
            $shapeC = $C->shape();
            if($N!=$shapeC[0] || $N!=$shapeC[1]) {
                throw new InvalidArgumentException('"C" rows and cols must have the same number of "A" cols');
            }
        } else {
            $C = $this->zeros($this->alloc([$N,$N],$A->dtype()));
        }
        $CC = $C->buffer();
        $offC = $C->offset();

        $lda = ($trans) ? $N : $K;
        $ldb = ($trans) ? $N : $K;
        $ldc = $N;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        return [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ];
    }

    public function translate_trmm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        ?bool $right=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ?bool $unit=null,
        ) : array
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($shapeA[0]!=$shapeA[1]) {
            throw new InvalidArgumentException('Matrix A must be square.: '.
                '['.implode(',',$shapeA).']');
        }
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeB).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $trans = $this->transToCode($trans,$conj);
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $order = BLAS::RowMajor;

        return [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ];
    }

    public function translate_trsm(
        NDArray $A,
        NDArray $B,
        float|object|null $alpha=null,
        ?bool $right=null,
        ?bool $lower=null,
        ?bool $trans=null,
        ?bool $conj=null,
        ?bool $unit=null,
        ) : array
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        $shapeA = $A->shape();
        $shapeB = $B->shape();
        if($right) {
            $sizeA = $shapeB[1];
        } else {
            $sizeA = $shapeB[0];
        }
        if($sizeA!=$shapeA[0]) {
            throw new InvalidArgumentException('Unmatch shape of Matrix A and B: '.
                '['.implode(',',$shapeA).'] <=> ['.implode(',',$shapeA).']');
        }
        $AA   = $A->buffer();
        $offA = $A->offset();
        $BB   = $B->buffer();
        $offB = $B->offset();
        $M = $shapeB[0];
        $N = $shapeB[1];

        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }

        $lda = ($right) ? $N : $M;
        $ldb = $N;
        $side  = ($right) ? BLAS::Right : BLAS::Left;
        $uplo  = ($lower) ? BLAS::Lower : BLAS::Upper;
        $diag  = ($unit)  ? BLAS::Unit  : BLAS::NonUnit;
        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        return [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ];

    }

    public function translate_omatcopy(
        NDArray $A,
        ?bool $trans=null,
        float|object|null $alpha=null,
        ?NDArray $B=null,
        ?bool $conj=null,
        ) : array
    {
        [$trans,$conj] = $this->complementTrans($trans,$conj,$A->dtype());

        if($A->ndim()!=2 || $B->ndim()!=2) {
            throw new InvalidArgumentException('Dimensions must be 2D-NDArray');
        }
        [$rows,$cols] = $A->shape();
        if($trans) {
            [$rows,$cols] = [$cols,$rows];
        }
        if($B->shape()!=[$rows,$cols]) {
            $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }

        [$M,$N] = $A->shape();
        if($alpha===null) {
            if($this->isComplex($A->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $N;
        $BB = $B->buffer();
        $offB = $B->offset();
        $ldB = $cols;

        $trans = $this->transToCode($trans,$conj);
        $order = BLAS::RowMajor;

        return [
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
        ];
    }

    public static function providerDtypesFloats()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
        ];
    }

    public function testGetNumThreads()
    {
        $blas = $this->getBlas();
        $n = $blas->getNumThreads();
        $this->assertGreaterThan(0,$n);
    }

    public function testGetNumProcs()
    {
        $blas = $this->getBlas();
        $n = $blas->getNumProcs();
        $this->assertGreaterThan(0,$n);
    }

    public function testGetConfig()
    {
        $blas = $this->getBlas();
        $s = $blas->getConfig();

        $this->assertTrue(
            strpos($s,'OpenBLAS')===0 ||
            strpos($s,'NO_LAPACKE')===0||
            strpos($s,'vecLib')===0
        );
    }

    public function testGetCorename()
    {
        $blas = $this->getBlas();
        $s = $blas->getCorename();
        $this->assertTrue(is_string($s));
    }

    public function testGetParallel()
    {
        $blas = $this->getBlas();
        $n = $blas->getParallel();
        $mt_enabled = ($n==OpenBLAS::OPENBLAS_THREAD || $n==OpenBLAS::OPENBLAS_OPENMP);
        $this->assertTrue($mt_enabled);
    }

    public function testComplexValue()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();
        $cval = $blas->getFFI()->new('openblas_complex_float');
        $this->assertTrue(is_a($cval,'FFI\CData'));
        $this->assertTrue($cval instanceof FFI\CData);
    }

    public function testScalNormal()
    {
        $blas = $this->getBlas();
        // float32
        $X = $this->array([1,2,3],dtype:NDArray::float32);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $blas->scal($N,$alpha,$XX,$offX,$incX);
        $this->assertEquals([2,4,6],$X->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
            [$N,$alpha,$XX,$offX,$incX] =
                $this->translate_scal(C(2),$X);

            $blas->scal($N,$alpha,$XX,$offX,$incX);
            //$this->assertEquals([2,4,6],$X->toArray());
            $this->assertEquals($this->toComplex([2,4,6]),$this->toComplex($X->toArray()));
        }

        if($this->fp64()) {
            // float64
            $dtype = NDArray::float64;
            $X = $this->array([1,2,3],dtype:$dtype);
            [$N,$alpha,$XX,$offX,$incX] =
                $this->translate_scal(2,$X);

            $blas->scal($N,$alpha,$XX,$offX,$incX);
            $this->assertEquals([2,4,6],$X->toArray());

            if(!$this->notSupportComplex()) {
                // complex128
                $dtype = NDArray::complex128;
                $X = $this->array($this->toComplex([1,2,3]),dtype:$dtype);
                [$N,$alpha,$XX,$offX,$incX] =
                    $this->translate_scal(C(2),$X);

                $blas->scal($N,$alpha,$XX,$offX,$incX);
                $this->assertEquals($this->toComplex([2,4,6]),$X->toArray());
            }

        }
    }

    public function testScalMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $N = 0;

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $offX = -1;

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('Argument #3 ($X) must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalOverflowBufferXwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalOverflowBufferXwithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testScalOverflowBufferXwithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX] =
            $this->translate_scal(2,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $blas->scal($N,$alpha,$XX,$offX,$incX);
    }

    public function testAxpyNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([1,2,3],dtype:NDArray::float32);
        $Y = $this->array([10,20,30],dtype:NDArray::float32);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([12,24,36],$Y->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
            $Y = $this->array([C(10),C(20),C(30)],dtype:NDArray::complex64);
            [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
                $this->translate_axpy($X,$Y,alpha:C(2));

            $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
            //$this->assertEquals([12,24,36],$Y->toArray());
            $this->assertEquals($this->toComplex([12,24,36]),$this->toComplex($Y->toArray()));
        }

        if($this->fp64()) {
            // float64
            $dtype = NDArray::float64;
            $x = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
            $y = $this->array([[10,20,30],[40,50,60]],dtype:$dtype);
            [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] = 
                $this->translate_axpy($x,$y,2);
            $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);

            $this->assertEquals([[12,24,36],[48,60,72]],$y->toArray());

            if(!$this->notSupportComplex()) {
                // complex128
                $dtype = NDArray::complex128;
                $x = $this->array($this->toComplex([[1,2,3],[4,5,6]]),dtype:$dtype);
                $y = $this->array($this->toComplex([[10,20,30],[40,50,60]]),dtype:$dtype);
                [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] = 
                    $this->translate_axpy($x,$y,C(2));
                $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);

                $this->assertEquals($this->toComplex([[12,24,36],[48,60,72]]),$y->toArray());
            }
        }
    }

    public function testAxpyMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('Argument #3 ($X) must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferXwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferXwithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferXwithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyMinusOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $offY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than equals 0.');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyMinusIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyIllegalBufferY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferYwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $YY = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferXwithOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $offY = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testAxpyOverflowBufferYwithIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([10,20,30]);
        [$N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_axpy($X,$Y,alpha:2);

        $incY = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $blas->axpy($N,$alpha,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([1,2,3],dtype:NDArray::float32);
        $Y = $this->array([4,5,6],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(32,$dot);

        // float64
        $X = $this->array([1,2,3],dtype:NDArray::float64);
        $Y = $this->array([4,5,6],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(32,$dot);
    }

    public function testDotuNormal()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotu($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot));

        // complex128
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex128);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex128);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotu($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot));

        // complex64
        $X = $this->array([C(1,i:1),C(2,i:1),C(3,i:1)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotu($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(C(32,i:15),$this->toComplex($dot));

    }

    public function testDotuSubNormal()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        $dot = $this->array([C(0)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotuSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot[0]));

        // complex128
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex128);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex128);
        $dot = $this->array([C(0)],dtype:NDArray::complex128);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotuSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot[0]));

        // complex64
        $X = $this->array([C(1,i:1),C(2,i:1),C(3,i:1)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        $dot = $this->array([C(0)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotuSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals(C(32,i:15),$this->toComplex($dot[0]));

    }

    public function testDotcNormal()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotc($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot));

        // complex128
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex128);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex128);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotc($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot));

        // complex64
        $X = $this->array([C(1,i:1),C(2,i:1),C(3,i:1)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $dot = $blas->dotc($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(C(32,i:-15),$this->toComplex($dot));

    }

    public function testDotcSubNormal()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        $dot = $this->array([C(0)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotcSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot[0]));

        // complex128
        $X = $this->array([C(1),C(2),C(3)],dtype:NDArray::complex128);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex128);
        $dot = $this->array([C(0)],dtype:NDArray::complex128);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotcSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals($this->toComplex(32),$this->toComplex($dot[0]));

        // complex64
        $X = $this->array([C(1,i:1),C(2,i:1),C(3,i:1)],dtype:NDArray::complex64);
        $Y = $this->array([C(4),C(5),C(6)],dtype:NDArray::complex64);
        $dot = $this->array([C(0)],dtype:NDArray::complex64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);
        $RR = $dot->buffer();
        $offR = 0;

        $blas->dotcSub($N,$XX,$offX,$incX,$YY,$offY,$incY,$RR,$offR);
        $this->assertEquals(C(32,i:-15),$this->toComplex($dot[0]));

    }

    public function testDotMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferXwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferXwithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferXwithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotMinusOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $offY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than equals 0.');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotMinusIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotIllegalBufferY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferYwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $YY = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferXwithOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $offY = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotOverflowBufferYwithIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3]);
        $Y = $this->array([4,5,6]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_dot($X,$Y);

        $incY = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $dot = $blas->dot($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testDotu()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1,i:2),C(3,i:4)],dtype:NDArray::complex64);
        $Y = $this->array([C(5,i:6),C(7,i:8)],dtype:NDArray::complex64);

        [$N,$XX,$offX,$incX,$YY,$offY,$incY] = 
            $this->translate_dot($X,$Y);

        $dot = $blas->dotu($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(-18,$dot->real);
        $this->assertEquals(68,$dot->imag);


        // complex128
        $X = $this->array([C(1,i:2),C(3,i:4)],dtype:NDArray::complex128);
        $Y = $this->array([C(5,i:6),C(7,i:8)],dtype:NDArray::complex128);

        [$N,$XX,$offX,$incX,$YY,$offY,$incY] = 
            $this->translate_dot($X,$Y);

        $dot = $blas->dotu($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(-18,$dot->real);
        $this->assertEquals(68,$dot->imag);
    }

    public function testDotc()
    {
        if($this->skipComplex()) {
            return;
        }
        $blas = $this->getBlas();

        // complex64
        $X = $this->array([C(1,i:2),C(3,i:4)],dtype:NDArray::complex64);
        $Y = $this->array([C(5,i:6),C(7,i:8)],dtype:NDArray::complex64);

        [$N,$XX,$offX,$incX,$YY,$offY,$incY] = 
            $this->translate_dot($X,$Y);

        $dot = $blas->dotc($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(70,$dot->real);
        $this->assertEquals(-8,$dot->imag);


        // complex128
        $X = $this->array([C(1,i:2),C(3,i:4)],dtype:NDArray::complex128);
        $Y = $this->array([C(5,i:6),C(7,i:8)],dtype:NDArray::complex128);

        [$N,$XX,$offX,$incX,$YY,$offY,$incY] = 
            $this->translate_dot($X,$Y);

        $dot = $blas->dotc($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals(70,$dot->real);
        $this->assertEquals(-8,$dot->imag);
    }

    public function testAsumNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([100,-10,-1000],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->asum($N,$XX,$offX,$incX);
        $this->assertEquals(1110,$min);

        // float64
        $X = $this->array([100,-10,-1000],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->asum($N,$XX,$offX,$incX);
        $this->assertEquals(1110,$min);

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(100,i:0),C(-10,i:0),C(-1000,i:0)],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->asum($N,$XX,$offX,$incX);
            $this->assertEquals(1110,$min);

            // complex128
            $X = $this->array([C(100,i:0),C(-10,i:0),C(-1000,i:0)],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->asum($N,$XX,$offX,$incX);
            $this->assertEquals(1110,$min);
        }
    }

    public function testAsumMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumOverflowBufferXwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumOverflowBufferXwithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testAsumOverflowBufferXwithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->asum($N,$XX,$offX,$incX);
    }

    public function testNrm2Normal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([[1,2],[3,4]],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $nrm2 = $blas->nrm2($N,$XX,$offX,$incX);
        $true = sqrt(1+2**2+3**2+4**2);
        $this->assertLessThan(0.00001,abs($nrm2-$true));

        // float64
        $X = $this->array([[1,2],[3,4]],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $nrm2 = $blas->nrm2($N,$XX,$offX,$incX);
        $true = sqrt(1+2**2+3**2+4**2);
        $this->assertLessThan(0.00001,abs($nrm2-$true));

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([[C(1),C(2)],[C(3),C(4)]],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $nrm2 = $blas->nrm2($N,$XX,$offX,$incX);
            $true = sqrt(1+2**2+3**2+4**2);
            $this->assertLessThan(0.00001,abs($nrm2-$true));

            // complex128
            $X = $this->array([[C(1),C(2)],[C(3),C(4)]],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $nrm2 = $blas->nrm2($N,$XX,$offX,$incX);
            $true = sqrt(1+2**2+3**2+4**2);
            $this->assertLessThan(0.00001,abs($nrm2-$true));
        }
    }

    public function testAMaxNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([100,-10,1],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->iamax($N,$XX,$offX,$incX);
        $this->assertEquals(0,$min);

        // float64
        $X = $this->array([100,-10,1],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->iamax($N,$XX,$offX,$incX);
        $this->assertEquals(0,$min);

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(100),C(-10),C(1)],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->iamax($N,$XX,$offX,$incX);
            $this->assertEquals(0,$min);

            // complex128
            $X = $this->array([C(100),C(-10),C(1)],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->iamax($N,$XX,$offX,$incX);
            $this->assertEquals(0,$min);
        }
    }

    public function testAMaxMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxOverflowBufferXwithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxOverflowBufferXwithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAMaxOverflowBufferXwithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamax($N,$XX,$offX,$incX);
    }

    public function testAminNormal()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        // float32
        $X = $this->array([100,-10,1],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->iamin($N,$XX,$offX,$incX);
        $this->assertEquals(2,$min);

        // float64
        $X = $this->array([100,-10,1],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $min = $blas->iamin($N,$XX,$offX,$incX);
        $this->assertEquals(2,$min);

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(100),C(-10),C(1)],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->iamin($N,$XX,$offX,$incX);
            $this->assertEquals(2,$min);

            // complex128
            $X = $this->array([C(100),C(-10),C(1)],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX] =
                $this->translate_amin($X);

            $min = $blas->iamin($N,$XX,$offX,$incX);
            $this->assertEquals(2,$min);
        }
    }

    public function testAminMinusN()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminMinusOffsetX()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminMinusIncX()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminIllegalBufferX()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminOverflowBufferXwithSize()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminOverflowBufferXwithOffsetX()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testAminOverflowBufferXwithIncX()
    {
        if($this->skipiamin()) return;
        $blas = $this->getBlas();

        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for buffer');
        $min = $blas->iamin($N,$XX,$offX,$incX);
    }

    public function testCopyNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([100,10,1],dtype:NDArray::float32);
        $Y = $this->array([0,0,0],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([100,10,1],$Y->toArray());

        // float64
        $X = $this->array([100,10,1],dtype:NDArray::float64);
        $Y = $this->array([0,0,0],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([100,10,1],$Y->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(100),C(10),C(1)],dtype:NDArray::complex64);
            $Y = $this->array([C(0),C(0),C(0)],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
                $this->translate_copy($X,$Y);

            $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
            $this->assertEquals($this->toComplex([100,10,1]),$this->toComplex($Y->toArray()));

            // complex128
            $X = $this->array([C(100),C(10),C(1)],dtype:NDArray::complex128);
            $Y = $this->array([C(0),C(0),C(0)],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
                $this->translate_copy($X,$Y);

            $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
            $this->assertEquals($this->toComplex([100,10,1]),$this->toComplex($Y->toArray()));
        }
    }

    public function testCopyInteger()
    {
        $blas = $this->getBlas();

        $X = $this->array([1,2,3,4],dtype:NDArray::int32);
        $Y = $this->array([0,0,0,0],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,2,3,4],$Y->toArray());

        // incX=2 incY=1
        $X = $this->array([1,2,3,4],dtype:NDArray::int32);
        $Y = $this->array([0,0,0,0],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);
        $N = 2;
        $incX = 2;
        $incY = 1;
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,3,0,0],$Y->toArray());

        // incX=1 incY=2
        $X = $this->array([1,2,3,4],dtype:NDArray::int32);
        $Y = $this->array([0,0,0,0],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);
        $N = 2;
        $incX = 1;
        $incY = 2;
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,2,0],$Y->toArray());

        // incX=1 incY=2 offX=1 offY=0
        $X = $this->array([1,2,3,4],dtype:NDArray::int32);
        $Y = $this->array([0,0,0,0],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);
        $N = 2;
        $incX = 1;
        $incY = 2;
        $offX = 1;
        $offY = 0;
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([2,0,3,0],$Y->toArray());

        // incX=2 incY=1 offX=0 offY=1
        $X = $this->array([1,2,3,4],dtype:NDArray::int32);
        $Y = $this->array([0,0,0,0],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);
        $N = 2;
        $incX = 2;
        $incY = 1;
        $offX = 0;
        $offY = 1;
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,3,0],$Y->toArray());

    }

    public function testCopyMinusN()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyMinusOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyMinusIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyIllegalBufferX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyMinusOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $offY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than equals 0.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyMinusIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyIllegalBufferY()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferXWithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $XX = $this->array([100,10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferXWithOffsetX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferXWithIncX()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferYWithSize()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $YY = $this->array([100,10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferXWithOffsetY()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $offY = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testCopyOverflowBufferXWithIncY()
    {
        $blas = $this->getBlas();

        $X = $this->array([100,10,1]);
        $Y = $this->array([0,0,0]);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $incY = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY.');
        $blas->copy($N,$XX,$offX,$incX,$YY,$offY,$incY);
    }

    public function testSwapNormal()
    {
        $blas = $this->getBlas();

        // float32
        $X = $this->array([100,10,1],dtype:NDArray::float32);
        $Y = $this->array([200,20,2],dtype:NDArray::float32);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $blas->swap($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([200,20,2],$X->toArray());
        $this->assertEquals([100,10,1],$Y->toArray());

        // float64
        $X = $this->array([100,10,1],dtype:NDArray::float64);
        $Y = $this->array([200,20,2],dtype:NDArray::float64);
        [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
            $this->translate_copy($X,$Y);

        $blas->swap($N,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([200,20,2],$X->toArray());
        $this->assertEquals([100,10,1],$Y->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $X = $this->array([C(100),C(10),C(1)],dtype:NDArray::complex64);
            $Y = $this->array([C(200),C(20),C(2)],dtype:NDArray::complex64);
            [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
                $this->translate_copy($X,$Y);

            $blas->swap($N,$XX,$offX,$incX,$YY,$offY,$incY);
            $this->assertEquals($this->toComplex([200,20,2]),$this->toComplex($X->toArray()));
            $this->assertEquals($this->toComplex([100,10,1]),$this->toComplex($Y->toArray()));

            // complex128
            $X = $this->array([C(100),C(10),C(1)],dtype:NDArray::complex128);
            $Y = $this->array([C(200),C(20),C(2)],dtype:NDArray::complex128);
            [$N,$XX,$offX,$incX,$YY,$offY,$incY] =
                $this->translate_copy($X,$Y);

            $blas->swap($N,$XX,$offX,$incX,$YY,$offY,$incY);
            $this->assertEquals($this->toComplex([200,20,2]),$this->toComplex($X->toArray()));
            $this->assertEquals($this->toComplex([100,10,1]),$this->toComplex($Y->toArray()));
        }
    }

    public function testRotgNormal()
    {
        $blas = $this->getBlas();

        // float32
        $dtype = NDArray::float32;
        $inputs = [
            [1,1],
            [2,2],
            [3,3],
            [4,4],
            [5,5],
        ];
        foreach($inputs as [$xx,$yy]) {
            $X = $this->array($xx,dtype:$dtype);
            $Y = $this->array($yy,dtype:$dtype);
            [$AA,$offA,$BB,$offB,$CC,$offC,$SS,$offS] =
                $this->translate_rotg($X,$Y);
           
            $blas->rotg($AA,$offA,$BB,$offB,$CC,$offC,$SS,$offS);

            $rr = $AA[0];
            $zz = $BB[0];
            $cc = $CC[0];
            $ss = $SS[0];
            //echo "(x,y)=(".$X->buffer()[0].", ".$Y->buffer()[0].")\n";
            //echo "(r,z)=(".$rr.", ".$zz.")\n";
            //echo "(c,s)=(".$cc.", ".$ss.")\n";
            $this->assertLessThan(1e-7,abs($xx-$X->buffer()[0]));
            $this->assertLessThan(1e-7,abs($yy-$Y->buffer()[0]));
            $rx =  $cc * $xx + $ss * $yy;
            $ry = -$ss * $xx + $cc * $yy;
            #echo "(rx,ry)=(".$rx.",".$ry.")\n";
            $this->assertLessThan(1e-6,abs($rr-$rx));
            $this->assertLessThan(1e-6,abs(0-$ry));
        }

        // float64
        $dtype = NDArray::float64;
        $inputs = [
            [1,1],
            [2,2],
            [3,3],
            [4,4],
            [5,5],
        ];
        foreach($inputs as [$xx,$yy]) {
            $X = $this->array($xx,dtype:$dtype);
            $Y = $this->array($yy,dtype:$dtype);
            [$AA,$offA,$BB,$offB,$CC,$offC,$SS,$offS] =
                $this->translate_rotg($X,$Y);
           
            $blas->rotg($AA,$offA,$BB,$offB,$CC,$offC,$SS,$offS);

            $rr = $AA[0];
            $zz = $BB[0];
            $cc = $CC[0];
            $ss = $SS[0];
            //echo "(x,y)=(".$X->buffer()[0].", ".$Y->buffer()[0].")\n";
            //echo "(r,z)=(".$rr.", ".$zz.")\n";
            //echo "(c,s)=(".$cc.", ".$ss.")\n";
            $this->assertLessThan(1e-7,abs($xx-$X->buffer()[0]));
            $this->assertLessThan(1e-7,abs($yy-$Y->buffer()[0]));
            $rx =  $cc * $xx + $ss * $yy;
            $ry = -$ss * $xx + $cc * $yy;
            #echo "(rx,ry)=(".$rx.",".$ry.")\n";
            $this->assertLessThan(1e-6,abs($rr-$rx));
            $this->assertLessThan(1e-6,abs(0-$ry));
        }

    }
   
    public function testRotNormal()
    {
        $blas = $this->getBlas();

        // float32
        $dtype = NDArray::float32;
        $x = $this->array([1,2,3,4,5],dtype:$dtype);
        $y = $this->array([1,2,3,4,5],dtype:$dtype);
        $c = $this->array([cos(pi()/4)],dtype:$dtype);
        $s = $this->array([sin(pi()/4)],dtype:$dtype);

        [
            $N,
            $XX,$offX,$incX,$YY,$offY,$incY,
            $CC,$offC,$SS,$offS
        ] = $this->translate_rot($x,$y,$c,$s);

        $blas->rot(
            $N,
            $XX,$offX,$incX,$YY,$offY,$incY,
            $CC,$offC,$SS,$offS
        );
        for($i=0;$i<5;$i++) {
            $this->assertLessThan(1e-6,abs(sqrt(2)*($i+1)-$x->buffer()[$i]));
            $this->assertLessThan(1e-6,abs($y->buffer()[$i]));
        }
        [
            $N,
            $XX,$offX,$incX,$YY,$offY,$incY,
            $CC,$offC,$SS,$offS
        ] = $this->translate_rot($x,$y,$c,$s);

        $blas->rot(
            $N,
            $XX,$offX,$incX,$YY,$offY,$incY,
            $CC,$offC,$SS,$offS
        );
        for($i=0;$i<5;$i++) {
            $this->assertLessThan(1e-6,abs(($i+1)-$x->buffer()[$i]));
            $this->assertLessThan(1e-6,abs((-$i-1)-$y->buffer()[$i]));
        }
    }

    #[DataProvider('providerDtypesFloats')]
    public function testRotmgNormal($params)
    {
        extract($params);
        //echo "\n===========testRotmgNormal==============\n";
        $blas = $this->getBlas();

        $inputs = [
            [1,0],  [1,1],  [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1],
        ];
        $r = 1/sqrt(2);
        $trues = [
            [1,0],[$r,-$r],[0,-1],[-$r,-$r],[1,0], [$r,-$r],[0,-1],[-$r,-$r],
        ];
        $filter = [
            -1 => [ [1,1,1,1], [0,0, 0,0] ],
             0 => [ [0,1,1,0], [1,0, 0,1] ],
             1 => [ [1,0,0,1], [0,1,-1,0] ],
            -2 => [ [1,1,1,1], [1,0, 0,1] ],
        ];
        foreach($inputs as $idx => [$xx,$yy]) {
            $X = $this->array($xx,dtype:$dtype);
            $Y = $this->array($yy,dtype:$dtype);
            [$DD1,$offD1,$DD2,$offD2,$BB1,$offB1,$BB2,$offB2,$PP,$offP] =
                $this->translate_rotmg($X,$Y);
           
            $blas->rotmg($DD1,$offD1,$DD2,$offD2,$BB1,$offB1,$BB2,$offB2,$PP,$offP);

            $d1 = $DD1[0];
            $d2 = $DD2[0];
            $b1 = $BB1[0];
            $flag = $PP[0];
            $h = [$PP[1],$PP[2],$PP[3],$PP[4]];
            //echo "===========================\n";
            //echo "(x,y)=(".$xx.", ".$yy.")\n";
            //echo "(d1,d2)=(".$d1.", ".$d2.")\n";
            //echo "b1=".$b1."\n";
            //echo "flag=".$flag."\n";
            //echo "h=(".implode(',',$h).")\n";
            [$a,$b] = $filter[intval($flag)];
            for($i=0;$i<4;$i++) {
                $h[$i] = $h[$i]*$a[$i] + $b[$i];
            };
            //echo "hh=(".implode(',',$h).")\n";
            $h = $this->array($h,dtype:$dtype);
            $xy = $this->array([1,0],dtype:$dtype);
            $prd = $this->zeros([2],dtype:$dtype);
            $blas->gemv(
                BLAS::RowMajor,BLAS::NoTrans,
                2,2, // m,n
                1.0, // alpha
                $h->buffer(),$h->offset(),2,   // A
                $xy->buffer(),$xy->offset(),1, // x
                0.0,  // b
                $prd->buffer(),$prd->offset(),1, // c
            );
            $blas->scal(1,sqrt($d1),$prd->buffer(),0,1);
            $blas->scal(1,sqrt($d2),$prd->buffer(),1,1);
            //echo "prd=(".implode(',',$prd->toArray()).")\n";
            $trueVec = $this->array($trues[$idx],dtype:$dtype);
            //echo "trues=(".implode(',',$trueVec->toArray()).")\n";
            $blas->axpy(2,-1,$trueVec->buffer(),0,1,$prd->buffer(),0,1);
            $diff = $blas->asum(2,$prd->buffer(),0,1);
            //echo "diff=".$diff."\n";
            $this->assertLessThan(1e-7,$diff);
        }
    }
   
    #[DataProvider('providerDtypesFloats')]
    public function testRotmNormal($params)
    {
        extract($params);
        //echo "\n===========testRotmgNormal==============\n";
        $blas = $this->getBlas();

        $inputs = [
            [1,0],  [1,1],  [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1],
        ];
        $r = 1/sqrt(2);
        $trues = [
            [[  1, 0   ,-3   , 0   ],[  0, 2   , 0   ,-4   ]],
            [[ $r, 2*$r,-3*$r,-4*$r],[-$r, 2*$r, 3*$r,-4*$r]],
            [[  0, 2   , 0   ,-4   ],[ -1, 0   , 3   , 0   ]],
            [[-$r, 2*$r, 3*$r,-4*$r],[-$r,-2*$r, 3*$r, 4*$r]],
            [[  1, 0   ,-3   , 0   ],[  0, 2   , 0   ,-4   ]],
            [[ $r, 2*$r,-3*$r,-4*$r],[-$r, 2*$r, 3*$r,-4*$r]],
            [[  0, 2   , 0   ,-4   ],[ -1, 0   , 3   , 0   ]],
            [[-$r, 2*$r, 3*$r,-4*$r],[-$r,-2*$r, 3*$r, 4*$r]],
        ];
        foreach($inputs as $idx => [$xx,$yy]) {
            $X = $this->array($xx,dtype:$dtype);
            $Y = $this->array($yy,dtype:$dtype);
            [$DD1,$offD1,$DD2,$offD2,$BB1,$offB1,$BB2,$offB2,$PP,$offP] =
                $this->translate_rotmg($X,$Y);
           
            $blas->rotmg($DD1,$offD1,$DD2,$offD2,$BB1,$offB1,$BB2,$offB2,$PP,$offP);

            $d1 = $DD1[0];
            $d2 = $DD2[0];
            $b1 = $BB1[0];
            //echo "===========================\n";
            //echo "(x,y)=(".$xx.", ".$yy.")\n";
            //echo "(d1,d2)=(".$d1.", ".$d2.")\n";
            //echo "b1=".$b1."\n";
            //echo "flag=".$flag."\n";

            $x = $this->array([ 1, 0,-3, 0],dtype:$dtype);
            $y = $this->array([ 0, 2, 0,-4],dtype:$dtype);

            $prd = $this->zeros([2],dtype:$dtype);

            $blas->rotm(4,$x->buffer(),0,1,$y->buffer(),0,1,$PP,$offP);

            $blas->scal(4,sqrt($d1),$x->buffer(),0,1);
            $blas->scal(4,sqrt($d2),$y->buffer(),0,1);
            [$truesX,$truesY] = $trues[$idx];
            $truesX = $this->array($truesX,dtype:$dtype);
            $truesY = $this->array($truesY,dtype:$dtype);
            //echo "trues=(".implode(',',$trueX->toArray()).")\n";
            $blas->axpy(4,-1,$truesX->buffer(),0,1,$x->buffer(),0,1);
            $blas->axpy(4,-1,$truesY->buffer(),0,1,$y->buffer(),0,1);
            $diffX = $blas->asum(4,$x->buffer(),0,1);
            //echo "diffX=".$diffX."\n";
            $this->assertLessThan(1e-7,$diffX);
            $diffY = $blas->asum(4,$y->buffer(),0,1);
            //echo "diffX=".$diffX."\n";
            $this->assertLessThan(1e-7,$diffY);
        }
    }

    public function testGemvNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([[1,2,3],[4,5,6]],dtype:NDArray::float32);
        $X = $this->array([100,10,1],dtype:NDArray::float32);
        $Y = $this->ones([2],dtype:NDArray::float32);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);

        $this->assertEquals(
            [123,456]
        ,$Y->toArray());

        // float64
        $A = $this->array([[1,2,3],[4,5,6]],dtype:NDArray::float64);
        $X = $this->array([100,10,1],dtype:NDArray::float64);
        $Y = $this->ones([2],dtype:NDArray::float64);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);

        $this->assertEquals(
            [123,456]
        ,$Y->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([[1,2,3],[4,5,6]]),dtype:NDArray::complex64);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex64);
            $Y = $this->ones([2],dtype:NDArray::complex64);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([123,456]),
                $this->toComplex($Y->toArray())
            );

            // complex128
            $A = $this->array($this->toComplex([[1,2,3],[4,5,6]]),dtype:NDArray::complex128);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex128);
            $Y = $this->ones([2],dtype:NDArray::complex128);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([123,456]),
                $this->toComplex($Y->toArray())
            );

            // complex64 check imag
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:1),C(3,i:1)],
                [C(4),C(5),C(6)]
            ]),dtype:NDArray::complex64);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex64);
            $Y = $this->ones([2],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,alpha:$alpha,beta:$beta,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([C(123,i:111),C(456)]),
                $this->toComplex($Y->toArray())
            );

            // complex64 check imag has beta 1.0
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:1),C(3,i:1)],
                [C(4),C(5),C(6)]
            ]),dtype:NDArray::complex64);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex64);
            $Y = $this->ones([2],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = C(1.0);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,alpha:$alpha,beta:$beta,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([C(124,i:111),C(457)]),
                $this->toComplex($Y->toArray())
            );

            // complex64 check imag has beta 2.0
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:1),C(3,i:1)],
                [C(4),C(5),C(6)]
            ]),dtype:NDArray::complex64);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex64);
            $Y = $this->ones([2],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = C(2.0,i:1.0);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,alpha:$alpha,beta:$beta,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([C(125,i:112),C(458,i:1)]),
                $this->toComplex($Y->toArray())
            );

            // complex64 check imag has alpha 2.0
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:1),C(3,i:1)],
                [C(4),C(5),C(6)]
            ]),dtype:NDArray::complex64);
            $X = $this->array($this->toComplex([100,10,1]),dtype:NDArray::complex64);
            $Y = $this->ones([2],dtype:NDArray::complex64);
            $alpha = C(2.0,i:1.0);
            $beta  = null;

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,alpha:$alpha,beta:$beta,Y:$Y);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$ldA,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals(
                $this->toComplex([C(135,i:345),C(912,i:456)]),
                $this->toComplex($Y->toArray())
            );
        }

    }

    public function testGemvTranspose()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([10,1]);
        $Y = $this->zeros([3]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y,trans:true);

        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);

        $this->assertEquals(
            [14,25,36]
        ,$Y->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array([[C(1),C(2),C(3)],[C(4),C(5),C(6)]],dtype:NDArray::complex64);
            $X = $this->array([C(10),C(1)],dtype:NDArray::complex64);
            $Y = $this->zeros([3],dtype:NDArray::complex64);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$n,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,Y:$Y,trans:true);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$n,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals($this->toComplex(
                [14,25,36]
            ),$this->toComplex($Y->toArray()));

            // complex64 check trans and conj
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4,i:1),C(5,i:1),C(6,i:1)]],dtype:NDArray::complex64);
            $X = $this->array([C(10),C(1)],dtype:NDArray::complex64);
            $Y = $this->zeros([3],dtype:NDArray::complex64);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$n,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,Y:$Y,trans:true);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$n,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals($this->toComplex(
                [C(14,i:-11),C(25,i:-11),C(36,i:-11)]
            ),$this->toComplex($Y->toArray()));

            // complex64 check trans and no_conj
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4,i:1),C(5,i:1),C(6,i:1)]],dtype:NDArray::complex64);
            $X = $this->array([C(10),C(1)],dtype:NDArray::complex64);
            $Y = $this->zeros([3],dtype:NDArray::complex64);

            [ $order,$trans,$m,$n,$alpha,$AA,$offA,$n,
              $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                $this->translate_gemv($A,$X,Y:$Y,trans:true,conj:false);

            $blas->gemv(
                $order,$trans,
                $m,$n,
                $alpha,
                $AA,$offA,$n,
                $XX,$offX,$incX,
                $beta,
                $YY,$offY,$incY);

            $this->assertEquals($this->toComplex(
                [C(14,i:11),C(25,i:11),C(36,i:11)]
            ),$this->toComplex($Y->toArray()));

            // complex64 check no_trans and conj
            if(PHP_OS!='Darwin') {
                $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4,i:1),C(5,i:1),C(6,i:1)]],dtype:NDArray::complex64);
                $X = $this->array([C(100),C(10),C(1)],dtype:NDArray::complex64);
                $Y = $this->zeros([2],dtype:NDArray::complex64);
            
                [ $order,$trans,$m,$n,$alpha,$AA,$offA,$n,
                  $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
                    $this->translate_gemv($A,$X,Y:$Y,conj:true);
            
                $blas->gemv(
                    $order,$trans,
                    $m,$n,
                    $alpha,
                    $AA,$offA,$n,
                    $XX,$offX,$incX,
                    $beta,
                    $YY,$offY,$incY);
                
                $this->assertEquals($this->toComplex(
                    [C(123,i:-111),C(456,i:-111)]
                ),$this->toComplex($Y->toArray()));
            }
        }
    }

    public function testGemvMinusM()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $m = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusN()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusOffsetA()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than equals 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusLdA()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $ldA = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvIllegalBufferA()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusOffsetX()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than equals 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusIncX()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvIllegalBufferX()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusOffsetY()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $offY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than equals 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMinusIncY()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvIllegalBufferY()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvMatrixOverFlowNormal()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $AA = $this->array([1,2,3,4,5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvVectorXOverFlowNormal()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $XX = $this->array([10,1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemvVectorYOverFlowNormal()
    {
        $blas = $this->getBlas();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([100,10,1]);
        $Y = $this->zeros([2]);

        [ $order,$trans,$m,$n,$alpha,$AA,$offA,$ldA,
          $XX,$offX,$incX,$beta,$YY,$offY,$incY] =
            $this->translate_gemv($A,$X,Y:$Y);

        $YY = $this->array([0])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY');
        $blas->gemv(
            $order,$trans,
            $m,$n,
            $alpha,
            $AA,$offA,$ldA,
            $XX,$offX,$incX,
            $beta,
            $YY,$offY,$incY);
    }

    public function testGemmNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float32);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]],dtype:NDArray::float32);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->ones([3,3],dtype:NDArray::float32);
        $transA = false;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],$C->toArray());

        // float64
        $A = $this->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float64);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]],dtype:NDArray::float64);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->ones([3,3],dtype:NDArray::float64);
        $transA = false;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([[1,2,3],[4,5,6],[7,8,9]]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([[1,0,0],[0,1,0],[0,0,1]]),dtype:NDArray::complex64);
            $alpha = C(1.0);
            $beta  = C(0.0);
            $C = $this->ones([3,3],dtype:NDArray::complex64);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]),$this->toComplex($C->toArray()));

            // complex128
            $A = $this->array($this->toComplex([[1,2,3],[4,5,6],[7,8,9]]),dtype:NDArray::complex128);
            $B = $this->array($this->toComplex([[1,0,0],[0,1,0],[0,0,1]]),dtype:NDArray::complex128);
            $alpha = C(1.0);
            $beta  = C(0.0);
            $C = $this->ones([3,3],dtype:NDArray::complex128);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4),C(5),C(6)],[C(7),C(8),C(9)]],dtype:NDArray::complex64);
            $B = $this->array([[C(1,i:1),C(0,i:1),C(0,i:1)],[C(0),C(1),C(0)],[C(0),C(0),C(1)]],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->ones([3,3],dtype:NDArray::complex64);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(0,i:2),C(1,i:2),C(2,i:2)],
                [C(4,i:4),C(5,i:4),C(6,i:4)],
                [C(7,i:7),C(8,i:7),C(9,i:7)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag has beta 1.0
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4),C(5),C(6)],[C(7),C(8),C(9)]],dtype:NDArray::complex64);
            $B = $this->array([[C(1,i:1),C(0,i:1),C(0,i:1)],[C(0),C(1),C(0)],[C(0),C(0),C(1)]],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = C(1.0);
            $C = $this->ones([3,3],dtype:NDArray::complex64);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(1,i:2),C(2,i:2),C(3,i:2)],
                [C(5,i:4),C(6,i:4),C(7,i:4)],
                [C(8,i:7),C(9,i:7),C(10,i:7)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag has beta 2.0
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4),C(5),C(6)],[C(7),C(8),C(9)]],dtype:NDArray::complex64);
            $B = $this->array([[C(1,i:1),C(0,i:1),C(0,i:1)],[C(0),C(1),C(0)],[C(0),C(0),C(1)]],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = C(2.0,i:1.0);
            $C = $this->ones([3,3],dtype:NDArray::complex64);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(2,i:3),C(3,i:3),C(4,i:3)],
                [C(6,i:5),C(7,i:5),C(8,i:5)],
                [C(9,i:8),C(10,i:8),C(11,i:8)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag has alpha 2.0
            $A = $this->array([[C(1,i:1),C(2,i:1),C(3,i:1)],[C(4),C(5),C(6)],[C(7),C(8),C(9)]],dtype:NDArray::complex64);
            $B = $this->array([[C(1,i:1),C(0,i:1),C(0,i:1)],[C(0),C(1),C(0)],[C(0),C(0),C(1)]],dtype:NDArray::complex64);
            $alpha = C(2.0,i:1.0);
            $beta  = null;
            $C = $this->ones([3,3],dtype:NDArray::complex64);
            $transA = false;
            $transB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(-2,i:4),C(0,i:5),C(2,i:6)],
                [C(4,i:12),C(6,i:13),C(8,i:14)],
                [C(7,i:21),C(9,i:22),C(11,i:23)]
            ]),$this->toComplex($C->toArray()));
        }
    }

    public function testGemmTransposeSquareA()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float32);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]],dtype:NDArray::float32);
        $alpha = null;
        $beta  = null;
        $C = $this->zeros([3,3],dtype:NDArray::float32);
        $transA = true;
        $transB = null;
        $conjA = null;
        $conjB = null;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm(
                $A,$B,alpha:$alpha,beta:$beta,C:$C,
                transA:$transA,transB:$transB,
                conjA:$conjA,conjB:$conjB,
            );

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9]
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,0,0],
                [0,1,0],
                [0,0,1]
            ]),dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = true;
            $transB = null;
            $conjA = null;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [1,4,7],
                [2,5,8],
                [3,6,9]
            ]),$this->toComplex($C->toArray()));

            // complex64 check trans and conj
            $A = $this->array($this->toComplex([
                [C(1,i:9),C(2,i:8),C(3,i:7)],
                [C(4,i:6),C(5,i:5),C(6,i:4)],
                [C(7,i:3),C(8,i:2),C(9,i:1)]
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [C(1),C(0),C(0)],
                [C(0),C(1),C(0)],
                [C(0),C(0),C(1)]
            ]),dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = true;
            $transB = null;
            $conjA = null;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(1,i:-9),C(4,i:-6),C(7,i:-3)],
                [C(2,i:-8),C(5,i:-5),C(8,i:-2)],
                [C(3,i:-7),C(6,i:-4),C(9,i:-1)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check trans and no_conj
            $A = $this->array($this->toComplex([
                [C(1,i:9),C(2,i:8),C(3,i:7)],
                [C(4,i:6),C(5,i:5),C(6,i:4)],
                [C(7,i:3),C(8,i:2),C(9,i:1)]
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [C(1),C(0),C(0)],
                [C(0),C(1),C(0)],
                [C(0),C(0),C(1)]
            ]),dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = true;
            $transB = null;
            $conjA = false;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(1,i:9),C(4,i:6),C(7,i:3)],
                [C(2,i:8),C(5,i:5),C(8,i:2)],
                [C(3,i:7),C(6,i:4),C(9,i:1)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check no_trans and conj
            if(PHP_OS!='Darwin') {
                $A = $this->array($this->toComplex([
                    [C(1,i:9),C(2,i:8),C(3,i:7)],
                    [C(4,i:6),C(5,i:5),C(6,i:4)],
                    [C(7,i:3),C(8,i:2),C(9,i:1)]
                ]),dtype:NDArray::complex64);
                $B = $this->array($this->toComplex([
                    [C(1),C(0),C(0)],
                    [C(0),C(1),C(0)],
                    [C(0),C(0),C(1)]
                ]),dtype:NDArray::complex64);
                $alpha = null;
                $beta  = null;
                $C = $this->zeros([3,3],dtype:NDArray::complex64);
                $transA = false;
                $transB = null;
                $conjA = true;
                $conjB = null;
            
                [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
                  $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                    $this->translate_gemm(
                        $A,$B,alpha:$alpha,beta:$beta,C:$C,
                        transA:$transA,transB:$transB,
                        conjA:$conjA,conjB:$conjB,
                    );
                
                $blas->gemm(
                    $order,$transA,$transB,
                    $M,$N,$K,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb,
                    $beta,
                    $CC,$offC,$ldc);
                
                $this->assertEquals($this->toComplex([
                    [C(1,i:-9),C(2,i:-8),C(3,i:-7)],
                    [C(4,i:-6),C(5,i:-5),C(6,i:-4)],
                    [C(7,i:-3),C(8,i:-2),C(9,i:-1)]
                ]),$this->toComplex($C->toArray()));
            }


            // complex64 check alpha with trans and conj
            $A = $this->array($this->toComplex([
                [C(1,i:9),C(2,i:8),C(3,i:7)],
                [C(4,i:6),C(5,i:5),C(6,i:4)],
                [C(7,i:3),C(8,i:2),C(9,i:1)]
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [C(1),C(0),C(0)],
                [C(0),C(1),C(0)],
                [C(0),C(0),C(1)]
            ]),dtype:NDArray::complex64);
            $alpha = C(0,i:1);
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = true;
            $transB = null;
            $conjA = null;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(9,i:1),C(6,i:4),C(3,i:7)],
                [C(8,i:2),C(5,i:5),C(2,i:8)],
                [C(7,i:3),C(4,i:6),C(1,i:9)]
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testGemmTransposeSquareB()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]],dtype:NDArray::float32);
        $B = $this->array([[1,2,3],[4,5,6],[7,8,9]],dtype:NDArray::float32);
        $alpha = null;
        $beta  = null;
        $C = $this->zeros([3,3],dtype:NDArray::float32);
        $transA = null;
        $transB = true;
        $conjA = null;
        $conjB = null;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm(
                $A,$B,alpha:$alpha,beta:$beta,C:$C,
                transA:$transA,transB:$transB,
                conjA:$conjA,conjB:$conjB,
            );

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,4,7],
            [2,5,8],
            [3,6,9]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,0,0],[0,1,0],[0,0,1]
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,2,3],[4,5,6],[7,8,9]
            ]),dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = null;
            $transB = true;
            $conjA = null;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [1,4,7],
                [2,5,8],
                [3,6,9]
            ]),$this->toComplex($C->toArray()));

            // complex64 check trans and conj
            $A = $this->array([
                [C(1),C(0),C(0)],
                [C(0),C(1),C(0)],
                [C(0),C(0),C(1)]
            ],dtype:NDArray::complex64);
            $B = $this->array([
                [C(1,i:9),C(2,i:8),C(3,i:7)],
                [C(4,i:6),C(5,i:5),C(6,i:4)],
                [C(7,i:3),C(8,i:2),C(9,i:1)]
            ],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = null;
            $transB = true;
            $conjA = null;
            $conjB = null;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(1,i:-9),C(4,i:-6),C(7,i:-3)],
                [C(2,i:-8),C(5,i:-5),C(8,i:-2)],
                [C(3,i:-7),C(6,i:-4),C(9,i:-1)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check trans and no_conj
            $A = $this->array([
                [C(1),C(0),C(0)],
                [C(0),C(1),C(0)],
                [C(0),C(0),C(1)]
            ],dtype:NDArray::complex64);
            $B = $this->array([
                [C(1,i:9),C(2,i:8),C(3,i:7)],
                [C(4,i:6),C(5,i:5),C(6,i:4)],
                [C(7,i:3),C(8,i:2),C(9,i:1)]
            ],dtype:NDArray::complex64);
            $alpha = null;
            $beta  = null;
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $transA = null;
            $transB = true;
            $conjA = null;
            $conjB = false;

            [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
              $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                $this->translate_gemm(
                    $A,$B,alpha:$alpha,beta:$beta,C:$C,
                    transA:$transA,transB:$transB,
                    conjA:$conjA,conjB:$conjB,
                );

            $blas->gemm(
                $order,$transA,$transB,
                $M,$N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc);

            $this->assertEquals($this->toComplex([
                [C(1,i:9),C(4,i:6),C(7,i:3)],
                [C(2,i:8),C(5,i:5),C(8,i:2)],
                [C(3,i:7),C(6,i:4),C(9,i:1)]
            ]),$this->toComplex($C->toArray()));

            // complex64 check no_trans and conj
            if(PHP_OS!='Darwin') {
                $A = $this->array([
                    [C(1),C(0),C(0)],
                    [C(0),C(1),C(0)],
                    [C(0),C(0),C(1)]
                ],dtype:NDArray::complex64);
                $B = $this->array([
                    [C(1,i:9),C(2,i:8),C(3,i:7)],
                    [C(4,i:6),C(5,i:5),C(6,i:4)],
                    [C(7,i:3),C(8,i:2),C(9,i:1)]
                ],dtype:NDArray::complex64);
                $alpha = null;
                $beta  = null;
                $C = $this->zeros([3,3],dtype:NDArray::complex64);
                $transA = null;
                $transB = null;
                $conjA = null;
                $conjB = true;
            
                [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
                  $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
                    $this->translate_gemm(
                        $A,$B,alpha:$alpha,beta:$beta,C:$C,
                        transA:$transA,transB:$transB,
                        conjA:$conjA,conjB:$conjB,
                    );
                
                $blas->gemm(
                    $order,$transA,$transB,
                    $M,$N,$K,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb,
                    $beta,
                    $CC,$offC,$ldc);
                
                $this->assertEquals($this->toComplex([
                    [C(1,i:-9),C(2,i:-8),C(3,i:-7)],
                    [C(4,i:-6),C(5,i:-5),C(6,i:-4)],
                    [C(7,i:-3),C(8,i:-2),C(9,i:-1)]
                ]),$this->toComplex($C->toArray()));
            }
        }

    }

    public function testGemmNoTransRectangleA23()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2,3],[4,5,6]]);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,3]);
        $transA = false;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,2,3],
            [4,5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleA32()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4],[5,6]]);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,3]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,3,5],
            [2,4,6],
        ],$C->toArray());
    }

    public function testGemmNoTransRectangleB32()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $this->array([[1,2],[3,4],[5,6]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,2]);
        $transA = false;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,2],
            [3,4],
            [5,6],
        ],$C->toArray());
    }

    public function testGemmTransposeRectangleB23()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $this->array([[1,2,3],[4,5,6]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,2]);
        $transA = false;
        $transB = true;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);

        $this->assertEquals([
            [1,4],
            [2,5],
            [3,6],
        ],$C->toArray());
    }

    public function testGemmMinusM()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusN()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusK()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $K = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument k must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusOffsetA()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than equals 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusLdA()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $lda = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmIllegalBufferA()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusOffsetB()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $offB = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetB must be greater than equals 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusLdB()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $ldb = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldB must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmIllegalBufferB()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $BB = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusOffsetC()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $offC = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetC must be greater than equals 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMinusLdC()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $ldc = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldC must be greater than 0.');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmIllegalBufferC()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4]]);
        $B = $this->array([[1,2],[3,4]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,2]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $CC = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMatrixAOverFlowTransposeRectangleA32()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4],[5,6]]);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,3]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $AA = $this->array([1,2,3,4,5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMatrixBOverFlowTransposeRectangleA32()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4],[5,6]]);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,3]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $BB = $this->array([1,0,0, 0,1,0, 0,0])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferB');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmOutputOverFlowTransposeRectangleA32()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,2],[3,4],[5,6]]);
        $B = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([2,3]);
        $transA = true;
        $transB = false;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $CC = $this->zeros([5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferC');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMatrixAOverFlowTransposeRectangleB23()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $this->array([[1,2,3],[4,5,6]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,2]);
        $transA = false;
        $transB = true;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $AA = $this->array([1,0,0, 0,1,0, 0,0])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmMatrixBOverFlowTransposeRectangleB23()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $this->array([[1,2,3],[4,5,6]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,2]);
        $transA = false;
        $transB = true;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $BB = $this->array([1,2,3,4,5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferB');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testGemmOutputOverFlowTransposeRectangleB23()
    {
        $blas = $this->getBlas();
        $A = $this->array([[1,0,0],[0,1,0],[0,0,1]]);
        $B = $this->array([[1,2,3],[4,5,6]]);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,2]);
        $transA = false;
        $transB = true;

        [ $order,$transA,$transB,$M,$N,$K,$alpha,$AA,$offA,$lda,
          $BB,$offB,$ldb,$beta,$CC,$offC,$ldc] =
            $this->translate_gemm($A,$B,alpha:$alpha,beta:$beta,C:$C,transA:$transA,transB:$transB);

        $CC = $this->zeros([5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferC');
        $blas->gemm(
            $order,$transA,$transB,
            $M,$N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc);
    }

    public function testSymmNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [1,2,3],
            [2,4,5],
            [3,5,6],
        ],dtype:NDArray::float32);
        $B = $this->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ],dtype:NDArray::float32);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,4],dtype:NDArray::float32);

        [
            $order,$side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_symm($A,$B,alpha:$alpha,beta:$beta,C:$C);

        $blas->symm(
            $order,
            $side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [38, 44, 50, 56],
            [67, 78, 89,100],
            [82, 96,110,124]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,2,3],
                [2,4,5],
                [3,5,6],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
            ]),dtype:NDArray::complex64);
            $alpha = C(1.0);
            $beta  = C(0.0);
            $C = $this->zeros([3,4],dtype:NDArray::complex64);

            [
                $order,$side,$uplo,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_symm($A,$B,alpha:$alpha,beta:$beta,C:$C);

            $blas->symm(
                $order,
                $side,$uplo,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [38, 44, 50, 56],
                [67, 78, 89,100],
                [82, 96,110,124]
            ]),$this->toComplex($C->toArray()));
        }

        // float64
        $A = $this->array([
            [1,2,3],
            [2,4,5],
            [3,5,6],
        ],dtype:NDArray::float64);
        $B = $this->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ],dtype:NDArray::float64);
        $alpha = 1.0;
        $beta  = 0.0;
        $C = $this->zeros([3,4],dtype:NDArray::float64);

        [
            $order,$side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_symm($A,$B,alpha:$alpha,beta:$beta,C:$C);

        $blas->symm(
            $order,
            $side,$uplo,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [38, 44, 50, 56],
            [67, 78, 89,100],
            [82, 96,110,124]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex128
            $A = $this->array($this->toComplex([
                [1,2,3],
                [2,4,5],
                [3,5,6],
            ]),dtype:NDArray::complex128);
            $B = $this->array($this->toComplex([
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
            ]),dtype:NDArray::complex128);
            $alpha = C(1.0);
            $beta  = C(0.0);
            $C = $this->zeros([3,4],dtype:NDArray::complex128);

            [
                $order,$side,$uplo,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_symm($A,$B,alpha:$alpha,beta:$beta,C:$C);

            $blas->symm(
                $order,
                $side,$uplo,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [38, 44, 50, 56],
                [67, 78, 89,100],
                [82, 96,110,124]
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testSyrkNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [ 1, 2, 3],
            [ 4, 5, 6],
            [ 7, 8, 9],
            [10,11,12],
        ],dtype:NDArray::float32);
        $C = $this->zeros([4,4],dtype:NDArray::float32);

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syrk($A,C:$C);

        $blas->syrk(
            $order,
            $uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [14, 32, 50, 68],
            [ 0, 77,122,167],
            [ 0,  0,194,266],
            [ 0,  0,  0,365],
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [ 1, 2, 3],
                [ 4, 5, 6],
                [ 7, 8, 9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([4,4],dtype:NDArray::complex64);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,C:$C);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [14, 32, 50, 68],
                [ 0, 77,122,167],
                [ 0,  0,194,266],
                [ 0,  0,  0,365],
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag
            $A = $this->array($this->toComplex([
                [C( 1,i:1),C( 2,i:1),C( 3,i:1)],
                [C( 4,i:1),C( 5,i:1),C( 6,i:1)],
                [C( 7,i:1),C( 8,i:1),C( 9,i:1)],
                [C(10,i:1),C(11,i:1),C(12,i:1)],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([4,4],dtype:NDArray::complex64);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,C:$C);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [C(11,i:12),C(29,i:21),C( 47,i:30),C( 65,i:39)],
                [C( 0,i: 0),C(74,i:30),C(119,i:39),C(164,i:48)],
                [C( 0,i: 0),C( 0,i: 0),C(191,i:48),C(263,i:57)],
                [C( 0,i: 0),C( 0,i: 0),C(  0,i: 0),C(362,i:66)],
            ]),$this->toComplex($C->toArray()));
        }

        // float64
        $A = $this->array([
            [ 1, 2, 3],
            [ 4, 5, 6],
            [ 7, 8, 9],
            [10,11,12],
        ],dtype:NDArray::float64);
        $C = $this->zeros([4,4],dtype:NDArray::float64);

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syrk($A,C:$C);

        $blas->syrk(
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [14, 32, 50, 68],
            [ 0, 77,122,167],
            [ 0,  0,194,266],
            [ 0,  0,  0,365],
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex128
            $A = $this->array($this->toComplex([
                [ 1, 2, 3],
                [ 4, 5, 6],
                [ 7, 8, 9],
                [10,11,12],
            ]),dtype:NDArray::complex128);
            $C = $this->zeros([4,4],dtype:NDArray::complex128);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,C:$C);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [14, 32, 50, 68],
                [ 0, 77,122,167],
                [ 0,  0,194,266],
                [ 0,  0,  0,365],
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testSyrkTranspose()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:NDArray::float32);
        $C = $this->zeros([3,3],dtype:NDArray::float32);
        $lower=null;
        $trans=true;
        $conj=null; // *** conj is forced to false ***

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syrk($A,C:$C,
                lower:$lower,trans:$trans,conj:$conj);

        $blas->syrk(
            $order,
            $uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [166,188,210],
            [  0,214,240],
            [  0,  0,270],
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64 trans and no_conj
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $alpha=C(1);
            $beta =C(0);
            $lower=null;
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,alpha:$alpha,beta:$beta,C:$C,
                    lower:$lower,trans:$trans,conj:$conj);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [166,188,210],
                [  0,214,240],
                [  0,  0,270],
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag trans and no_conj
            $A = $this->array($this->toComplex([
                [C( 1,i:1),C( 2,i:1),C( 3,i:1)],
                [C( 4,i:1),C( 5,i:1),C( 6,i:1)],
                [C( 7,i:1),C( 8,i:1),C( 9,i:1)],
                [C(10,i:1),C(11,i:1),C(12,i:1)],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $alpha=C(1);
            $beta =C(0);
            $lower=null;
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,alpha:$alpha,beta:$beta,C:$C,
                    lower:$lower,trans:$trans,conj:$conj);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [C(162,i:44),C(184,i:48),C(206,i:52)],
                [C(  0,i: 0),C(210,i:52),C(236,i:56)],
                [C(  0,i: 0),C(  0,i: 0),C(266,i:60)],
            ]),$this->toComplex($C->toArray()));

            // complex64 check alpha with trans and no_conj
            $A = $this->array($this->toComplex([
                [C( 1,i:12),C( 2,i:11),C( 3,i:10)],
                [C( 4,i: 9),C( 5,i: 8),C( 6,i: 7)],
                [C( 7,i: 6),C( 8,i: 5),C( 9,i: 4)],
                [C(10,i: 3),C(11,i: 2),C(12,i: 1)],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $alpha=C(0,i:1);
            $beta =C(0);
            $lower=null;
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syrk($A,alpha:$alpha,beta:$beta,C:$C,
                    lower:$lower,trans:$trans,conj:$conj);

            $blas->syrk(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [C(-240,i:-104),C(-248,i:-52),C(-256,i:  0)],
                [C(   0,i:   0),C(-248,i:  0),C(-248,i: 52)],
                [C(   0,i:   0),C(   0,i:  0),C(-240,i:104)],
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testSyr2kNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:NDArray::float32);
        $B = $this->array([
            [1,3,5],
            [2,4,6],
            [7,9,11],
            [8,10,12],
        ],dtype:NDArray::float32);
        $C = $this->zeros([4,4],dtype:NDArray::float32);

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syr2k($A,$B,C:$C);

        $blas->syr2k(
            $order,
            $uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [44, 77,134,167],
            [ 0,128,239,290],
            [ 0,  0,440,545],
            [ 0,  0,  0,668]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,3,5],
                [2,4,6],
                [7,9,11],
                [8,10,12],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([4,4],dtype:NDArray::complex64);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syr2k($A,$B,C:$C);

            $blas->syr2k(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [44, 77,134,167],
                [ 0,128,239,290],
                [ 0,  0,440,545],
                [ 0,  0,  0,668]
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:1),C(3,i:1)],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,3,5],
                [2,4,6],
                [7,9,11],
                [8,10,12],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([4,4],dtype:NDArray::complex64);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syr2k($A,$B,C:$C);

            $blas->syr2k(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [C(44,i:18),C( 77,i:12),C(134,i:27),C(167,i:30)],
                [C( 0,i: 0),C(128,i: 0),C(239,i: 0),C(290,i: 0)],
                [C( 0,i: 0),C(  0,i: 0),C(440,i: 0),C(545,i: 0)],
                [C( 0,i: 0),C(  0,i: 0),C(  0,i: 0),C(668,i: 0)],
            ]),$this->toComplex($C->toArray()));
        }

        // float64
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:NDArray::float64);
        $B = $this->array([
            [1,3,5],
            [2,4,6],
            [7,9,11],
            [8,10,12],
        ],dtype:NDArray::float64);
        $C = $this->zeros([4,4],dtype:NDArray::float64);

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syr2k($A,$B,C:$C);

        $blas->syr2k(
            $order,
            $uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [44, 77,134,167],
            [ 0,128,239,290],
            [ 0,  0,440,545],
            [ 0,  0,  0,668]
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex128
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex128);
            $B = $this->array($this->toComplex([
                [1,3,5],
                [2,4,6],
                [7,9,11],
                [8,10,12],
            ]),dtype:NDArray::complex128);
            $C = $this->zeros([4,4],dtype:NDArray::complex128);

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syr2k($A,$B,C:$C);

            $blas->syr2k(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [44, 77,134,167],
                [ 0,128,239,290],
                [ 0,  0,440,545],
                [ 0,  0,  0,668]
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testSyr2kTranspose()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:NDArray::float32);
        $B = $this->array([
            [1,3,5],
            [2,4,6],
            [7,9,11],
            [8,10,12],
        ],dtype:NDArray::float32);
        $C = $this->zeros([3,3],dtype:NDArray::float32);
        $lower=null;
        $trans=true;
        $conj=null; // *** conj is forced to false ***

        [
            $order,$uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        ] = $this->translate_syr2k($A,$B,C:$C,
                lower:$lower,trans:$trans,conj:$conj);

        $blas->syr2k(
            $order,
            $uplo,$trans,
            $N,$K,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb,
            $beta,
            $CC,$offC,$ldc
        );

        $this->assertEquals([
            [276,338,400],
            [  0,416,494],
            [  0,  0,588],
        ],$C->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1,3,5],
                [2,4,6],
                [7,9,11],
                [8,10,12],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $lower=null;
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syr2k($A,$B,C:$C,
                    lower:$lower,trans:$trans,conj:$conj);

            $blas->syr2k(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );


            $this->assertEquals($this->toComplex([
                [276,338,400],
                [  0,416,494],
                [  0,  0,588],
            ]),$this->toComplex($C->toArray()));

            // complex64 check imag trans and conj
            $A = $this->array($this->toComplex([
                [C( 1,i:12),C( 2,i:11),C( 3,i:10)],
                [C( 4,i: 9),C( 5,i: 8),C( 6,i: 7)],
                [C( 7,i: 6),C( 8,i: 5),C( 9,i: 4)],
                [C(10,i: 3),C(11,i: 2),C(12,i: 1)],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [C(1,i:1),C( 3,i:1),C( 5,i:1)],
                [C(2,i:1),C( 4,i:1),C( 6,i:1)],
                [C(7,i:1),C( 9,i:1),C(11,i:1)],
                [C(8,i:1),C(10,i:1),C(12,i:1)],
            ]),dtype:NDArray::complex64);
            $C = $this->zeros([3,3],dtype:NDArray::complex64);
            $lower=null;
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            ] = $this->translate_syr2k($A,$B,C:$C,
                    lower:$lower,trans:$trans,conj:$conj);

            $blas->syr2k(
                $order,
                $uplo,$trans,
                $N,$K,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb,
                $beta,
                $CC,$offC,$ldc
            );

            $this->assertEquals($this->toComplex([
                [C(216,i:236),C(282,i:282),C(348,i:328)],
                [C(  0,i:  0),C(364,i:312),C(446,i:342)],
                [C(  0,i:  0),C(  0,i:  0),C(544,i:356)],
            ]),$this->toComplex($C->toArray()));
        }

    }

    public function testTrmmNormal()
    {
        $blas = $this->getBlas();

        // float32
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:NDArray::float32);
        $B = $this->array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12],
        ],dtype:NDArray::float32);

        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trmm($A,$B);

        $blas->trmm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );

        $this->assertEquals([
            [ 38, 44, 50, 56],
            [ 65, 74, 83, 92],
            [ 54, 60, 66, 72]
        ],$B->toArray());

        if(!$this->notSupportComplex()) {
            // complex64
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:NDArray::complex64);

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [ 38, 44, 50, 56],
                [ 65, 74, 83, 92],
                [ 54, 60, 66, 72]
            ]),$this->toComplex($B->toArray()));

            // complex64 check imag
            $A = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(9,i:0),C(4,i:3),C(5,i:2)],
                [C(9,i:0),C(9,i:0),C(6,i:1)],
            ]),dtype:NDArray::complex64);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:NDArray::complex64);

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [C(38,i:67),C(44,i:82),C(50,i:97),C(56,i:112)],
                [C(65,i:33),C(74,i:38),C(83,i:43),C(92,i: 48)],
                [C(54,i: 9),C(60,i:10),C(66,i:11),C(72,i: 12)]
            ]),$this->toComplex($B->toArray()));
        }

        // float64
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:NDArray::float64);
        $B = $this->array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12],
        ],dtype:NDArray::float64);

        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trmm($A,$B);

        $blas->trmm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );

        $this->assertEquals([
            [ 38, 44, 50, 56],
            [ 65, 74, 83, 92],
            [ 54, 60, 66, 72]
        ],$B->toArray());

        if(!$this->notSupportComplex()) {
            // complex128
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:NDArray::complex128);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:NDArray::complex128);

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [ 38, 44, 50, 56],
                [ 65, 74, 83, 92],
                [ 54, 60, 66, 72]
            ]),$this->toComplex($B->toArray()));
        }

    }

    public function testTrmmTranspose()
    {
        $blas = $this->getBlas();

        // float32 trans
        $dtype = NDArray::float32; 
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:$dtype);
        $B = $this->array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12],
        ],dtype:$dtype);
        $trans=true;
        $conj=null; // *** conj is forced to false ***

        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trmm($A,$B,trans:$trans,conj:$conj);

        $blas->trmm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );

        $this->assertEquals([
            [  1,  2,  3,  4],
            [ 22, 28, 34, 40],
            [ 82, 96,110,124]
        ],$B->toArray());

        if(!$this->notSupportComplex()) {
            // complex64 trans
            $dtype = NDArray::complex64; 
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:$dtype);
            $trans=true;
            $conj=null; // *** conj is forced to false ***

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B,trans:$trans,conj:$conj);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [  1,  2,  3,  4],
                [ 22, 28, 34, 40],
                [ 82, 96,110,124]
            ]),$this->toComplex($B->toArray()));

            // complex64 check imag trans and conj
            $dtype = NDArray::complex64; 
            $A = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(9,i:0),C(4,i:3),C(5,i:2)],
                [C(9,i:0),C(9,i:0),C(6,i:1)],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:$dtype);
            $trans=true;
            $conj=null;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B,trans:$trans,conj:$conj);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [C( 1,i: -6),C( 2,i:-12),C(  3,i:-18),C(  4,i:-24)],
                [C(22,i:-20),C(28,i:-28),C( 34,i:-36),C( 40,i:-44)],
                [C(82,i:-23),C(96,i:-30),C(110,i:-37),C(124,i:-44)]
            ]),$this->toComplex($B->toArray()));

            // complex64 check imag trans and no_conj
            $dtype = NDArray::complex64; 
            $A = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(9,i:0),C(4,i:3),C(5,i:2)],
                [C(9,i:0),C(9,i:0),C(6,i:1)],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9,10,11,12],
            ]),dtype:$dtype);
            $trans=true;
            $conj=false;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trmm($A,$B,trans:$trans,conj:$conj);

            $blas->trmm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );

            $this->assertEquals($this->toComplex([
                [C( 1,i: 6),C( 2,i:12),C(  3,i:18),C(  4,i:24)],
                [C(22,i:20),C(28,i:28),C( 34,i:36),C( 40,i:44)],
                [C(82,i:23),C(96,i:30),C(110,i:37),C(124,i:44)]
            ]),$this->toComplex($B->toArray()));

            // complex64 check imag no_trans and conj
            if(PHP_OS!='Darwin') {
                $dtype = NDArray::complex64; 
                $A = $this->array($this->toComplex([
                    [C(1,i:6),C(2,i:5),C(3,i:4)],
                    [C(9,i:0),C(4,i:3),C(5,i:2)],
                    [C(9,i:0),C(9,i:0),C(6,i:1)],
                ]),dtype:$dtype);
                $B = $this->array($this->toComplex([
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9,10,11,12],
                ]),dtype:$dtype);
                $trans=false;
                $conj=true;
            
                [
                    $order,$side,$uplo,$trans,$diag,
                    $M,$N,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb
                ] = $this->translate_trmm($A,$B,trans:$trans,conj:$conj);
                
                $blas->trmm(
                    $order,$side,$uplo,$trans,$diag,
                    $M,$N,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb
                );
            
                $this->assertEquals($this->toComplex([
                    [C(38,i:-67),C(44,i:-82),C(50,i:-97),C(56,i:-112)],
                    [C(65,i:-33),C(74,i:-38),C(83,i:-43),C(92,i: -48)],
                    [C(54,i: -9),C(60,i:-10),C(66,i:-11),C(72,i: -12)]
                ]),$this->toComplex($B->toArray()));
            }
        }

    }

    public function testTrsmNormal()
    {
        $blas = $this->getBlas();

        // float32
        $dtype = NDArray::float32;
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:$dtype);
        $B = $this->array([
            [ 7, 8],
            [10,11],
            [13,14],
        ],dtype:$dtype);
        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trsm($A,$B);
        $origB = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

        $blas->trsm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );
        $RA = $this->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ],dtype:$A->dtype());
        $C = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->gemm(
            BLAS::RowMajor,$trans,BLAS::NoTrans,
            $M,$N,$M,
            1.0,
            $RA->buffer(),0,$M,
            $BB,$offB,$ldb,
            0.0,
            $C->buffer(),0,$N
        );

        $this->assertTrue($this->isclose($C,$origB));

        if(!$this->notSupportComplex()) {
            // complex64
            $dtype = NDArray::complex64;
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [ 7, 8],
                [10,11],
                [13,14],
            ]),dtype:$dtype);
            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [1,2,3],
                [0,4,5],
                [0,0,6],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                C(1.0),
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            //echo "=====C=====\n";
            //$CC = $C->buffer();
            //$len = count($CC);
            //for($i=0;$i<$len;++$i) {
            //    echo $CC[$i]->real.','.$CC[$i]->imag."\n";
            //}

            $this->assertTrue($this->isclose($C,$origB,atol:1e-6));
        }

        // float64
        $dtype = NDArray::float64;
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:$dtype);
        $B = $this->array([
            [ 7, 8],
            [10,11],
            [13,14],
        ],dtype:$dtype);
        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trsm($A,$B);
        $origB = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

        $blas->trsm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );
        $RA = $this->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ],dtype:$A->dtype());
        $C = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->gemm(
            BLAS::RowMajor,$trans,BLAS::NoTrans,
            $M,$N,$M,
            1.0,
            $RA->buffer(),0,$M,
            $BB,$offB,$ldb,
            0.0,
            $C->buffer(),0,$N
        );

        //echo "=====realC=====\n";
        //$CC = $C->buffer();
        //$len = count($CC);
        //for($i=0;$i<$len;++$i) {
        //    echo $CC[$i]."\n";
        //}
        $this->assertTrue($this->isclose($C,$origB));

        if(!$this->notSupportComplex()) {
            // complex128
            $dtype = NDArray::complex128;
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [ 7, 8],
                [10,11],
                [13,14],
            ]),dtype:$dtype);
            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [1,2,3],
                [0,4,5],
                [0,0,6],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                C(1.0),
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            //echo "=====C=====\n";
            //$CC = $C->buffer();
            //$len = count($CC);
            //for($i=0;$i<$len;++$i) {
            //    echo $CC[$i]->real.','.$CC[$i]->imag."\n";
            //}

            $this->assertTrue($this->isclose($C,$origB,atol:1e-6));
        }
    }

    public function testTrsmTranspose()
    {
        $blas = $this->getBlas();

        if(!$this->notSupportComplex()) {
            // complex64 check imag trans and conj
            $dtype = NDArray::complex64;
            $A = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:0)],
                [C(9,i:9),C(3,i:0)],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [C( 4,i:0)],
                [C( 6,i:0)],
            ]),dtype:$dtype);
            $trans = true;
            $conj  = null;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [C(1,i:1),C(2,i:0)],
                [C(0,i:0),C(3,i:0)],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                $alpha,
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            //echo "=====B=====\n";
            //$BB = $B->buffer();
            //$len = count($BB);
            //for($i=0;$i<$len;++$i) {
            //    echo $BB[$i]->real.','.$BB[$i]->imag."\n";
            //}
            //echo "=====C=====\n";
            //$CC = $C->buffer();
            //$len = count($CC);
            //for($i=0;$i<$len;++$i) {
            //    echo $CC[$i]->real.','.$CC[$i]->imag."\n";
            //}

            $this->assertTrue($this->isclose($C,$origB));
        }

        // float32
        $dtype = NDArray::float32;
        $A = $this->array([
            [1,2,3],
            [9,4,5],
            [9,9,6],
        ],dtype:$dtype);
        $B = $this->array([
            [ 7, 8],
            [10,11],
            [13,14],
        ],dtype:$dtype);
        $trans = true;
        $conj  = null;

        [
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
        $origB = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);
        $blas->trsm(
            $order,$side,$uplo,$trans,$diag,
            $M,$N,
            $alpha,
            $AA,$offA,$lda,
            $BB,$offB,$ldb
        );
        $RA = $this->array([
            [1,2,3],
            [0,4,5],
            [0,0,6],
        ],dtype:$A->dtype());
        $C = $this->zeros($B->shape(),dtype:$B->dtype());
        $blas->gemm(
            BLAS::RowMajor,$trans,BLAS::NoTrans,
            $M,$N,$M,
            1.0,
            $RA->buffer(),0,$M,
            $BB,$offB,$ldb,
            0.0,
            $C->buffer(),0,$N
        );

        $this->assertTrue($this->isclose($C,$origB));

        if(!$this->notSupportComplex()) {
            // complex64
            $dtype = NDArray::complex64;
            $A = $this->array($this->toComplex([
                [1,2,3],
                [9,4,5],
                [9,9,6],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [ 7, 8],
                [10,11],
                [13,14],
            ]),dtype:$dtype);
            $trans = true;
            $conj  = null;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);
            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [1,2,3],
                [0,4,5],
                [0,0,6],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                $alpha,
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            $this->assertTrue($this->isclose($C,$origB));

            // complex64 check imag trans and conj
            $dtype = NDArray::complex64;
            $A = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(9,i:9),C(4,i:3),C(5,i:2)],
                [C(9,i:9),C(9,i:9),C(6,i:1)],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [C( 7,i:1),C( 8,i:2)],
                [C(10,i:1),C(11,i:2)],
                [C(13,i:1),C(14,i:2)],
            ]),dtype:$dtype);
            $trans = true;
            $conj  = null;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(0,i:0),C(4,i:3),C(5,i:2)],
                [C(0,i:0),C(0,i:0),C(6,i:1)],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                $alpha,
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            //echo "=====B=====\n";
            //$BB = $B->buffer();
            //$len = count($BB);
            //for($i=0;$i<$len;++$i) {
            //    echo $BB[$i]->real.','.$BB[$i]->imag."\n";
            //}
            //echo "=====C=====\n";
            //$CC = $C->buffer();
            //$len = count($CC);
            //for($i=0;$i<$len;++$i) {
            //    echo $CC[$i]->real.','.$CC[$i]->imag."\n";
            //}

            $this->assertTrue($this->isclose($C,$origB));

            // complex64 check imag trans and no_conj
            $dtype = NDArray::complex64;
            $A = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(9,i:9),C(4,i:3),C(5,i:2)],
                [C(9,i:9),C(9,i:9),C(6,i:1)],
            ]),dtype:$dtype);
            $B = $this->array($this->toComplex([
                [C( 7,i:1),C( 8,i:2)],
                [C(10,i:1),C(11,i:2)],
                [C(13,i:1),C(14,i:2)],
            ]),dtype:$dtype);
            $trans = true;
            $conj  = false;

            [
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
            $origB = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);

            $blas->trsm(
                $order,$side,$uplo,$trans,$diag,
                $M,$N,
                $alpha,
                $AA,$offA,$lda,
                $BB,$offB,$ldb
            );
            $RA = $this->array($this->toComplex([
                [C(1,i:6),C(2,i:5),C(3,i:4)],
                [C(0,i:0),C(4,i:3),C(5,i:2)],
                [C(0,i:0),C(0,i:0),C(6,i:1)],
            ]),dtype:$A->dtype());
            $C = $this->zeros($B->shape(),dtype:$B->dtype());
            $blas->gemm(
                BLAS::RowMajor,$trans,BLAS::NoTrans,
                $M,$N,$M,
                $alpha,
                $RA->buffer(),0,$M,
                $BB,$offB,$ldb,
                C(0.0),
                $C->buffer(),0,$N
            );

            $this->assertTrue($this->isclose($C,$origB));

            // complex64 check imag no_trans and conj
            if(PHP_OS!='Darwin') {
                $dtype = NDArray::complex64;
                $A = $this->array($this->toComplex([
                    [C(1,i:6),C(2,i:5),C(3,i:4)],
                    [C(9,i:9),C(4,i:3),C(5,i:2)],
                    [C(9,i:9),C(9,i:9),C(6,i:1)],
                ]),dtype:$dtype);
                $B = $this->array($this->toComplex([
                    [C( 7,i:1),C( 8,i:2)],
                    [C(10,i:1),C(11,i:2)],
                    [C(13,i:1),C(14,i:2)],
                ]),dtype:$dtype);
                $trans = false;
                $conj  = true;
            
                [
                    $order,$side,$uplo,$trans,$diag,
                    $M,$N,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb
                ] = $this->translate_trsm($A,$B,trans:$trans,conj:$conj);
                $origB = $this->zeros($B->shape(),dtype:$B->dtype());
                $blas->copy(count($BB),$BB,0,1,$origB->buffer(),0,1);
                
                $blas->trsm(
                    $order,$side,$uplo,$trans,$diag,
                    $M,$N,
                    $alpha,
                    $AA,$offA,$lda,
                    $BB,$offB,$ldb
                );
                $RA = $this->array($this->toComplex([
                    [C(1,i:6),C(2,i:5),C(3,i:4)],
                    [C(0,i:0),C(4,i:3),C(5,i:2)],
                    [C(0,i:0),C(0,i:0),C(6,i:1)],
                ]),dtype:$A->dtype());
                $C = $this->zeros($B->shape(),dtype:$B->dtype());
                $blas->gemm(
                    BLAS::RowMajor,$trans,BLAS::NoTrans,
                    $M,$N,$M,
                    $alpha,
                    $RA->buffer(),0,$M,
                    $BB,$offB,$ldb,
                    C(0.0),
                    $C->buffer(),0,$N
                );
            
                $this->assertTrue($this->isclose($C,$origB));
            }
        }

    }

    #[RequiresOperatingSystem('WINNT|Linux')]
    public function testOmatcopyNormal()
    {
        $blas = $this->getBlas();

        // float32
        $dtype = NDArray::float32;
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:$dtype);
        $B = $this->zeros([3,4],dtype:$dtype);

        [
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
        ] = $this->translate_omatcopy($A,trans:true,B:$B);

        $blas->omatcopy(
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
        );

        $this->assertEquals([
            [1,4,7,10],
            [2,5,8,11],
            [3,6,9,12],
        ],$B->toArray());

        if(!$this->notSupportComplex()) {
            // complex64 trans and conj
            $A = $this->array($this->toComplex([
                [C(1,i:100),C(2,i:200),3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->zeros([3,4],dtype:NDArray::complex64);

            [
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            ] = $this->translate_omatcopy($A,trans:true,B:$B);

            $blas->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );

            $this->assertEquals($this->toComplex([
                [C(1,i:-100),4,7,10],
                [C(2,i:-200),5,8,11],
                [3,6,9,12],
            ]),$this->toComplex($B->toArray()));

            // complex64  trans and no_conj
            $A = $this->array($this->toComplex([
                [C(1,i:100),C(2,i:200),3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->zeros([3,4],dtype:NDArray::complex64);

            [
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            ] = $this->translate_omatcopy($A,trans:true,conj:false,B:$B);
            $blas->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );

            $this->assertEquals($this->toComplex([
                [C(1,i:100),4,7,10],
                [C(2,i:200),5,8,11],
                [3,6,9,12],
            ]),$this->toComplex($B->toArray()));


            // complex64  no_trans and conj
            $A = $this->array($this->toComplex([
                [C(1,i:100),C(2,i:200),3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:NDArray::complex64);
            $B = $this->zeros([4,3],dtype:NDArray::complex64);

            [
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            ] = $this->translate_omatcopy($A,trans:false,conj:true,B:$B);
            $blas->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );

            $this->assertEquals($this->toComplex([
                [C(1,i:-100),C(2,i:-200),3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),$this->toComplex($B->toArray()));
        }

        // float64
        $type = NDArray::float64;
        $A = $this->array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
        ],dtype:$type);
        $B = $this->zeros([3,4],dtype:$type);

        [
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
        ] = $this->translate_omatcopy($A,trans:true,B:$B);

        $blas->omatcopy(
            $order,$trans,
            $M,$N,
            $alpha,
            $AA, $offA, $ldA,
            $BB, $offB, $ldB,
        );

        $this->assertEquals([
            [1,4,7,10],
            [2,5,8,11],
            [3,6,9,12],
        ],$B->toArray());

        if(!$this->notSupportComplex()) {
            // complex128
            $type = NDArray::complex128;
            $A = $this->array($this->toComplex([
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ]),dtype:$type);
            $B = $this->zeros([3,4],dtype:$type);

            [
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            ] = $this->translate_omatcopy($A,trans:true,B:$B);

            $blas->omatcopy(
                $order,$trans,
                $M,$N,
                $alpha,
                $AA, $offA, $ldA,
                $BB, $offB, $ldB,
            );

            $this->assertEquals($this->toComplex([
                [1,4,7,10],
                [2,5,8,11],
                [3,6,9,12],
            ]),$this->toComplex($B->toArray()));
        }
    }

}
