<?php
namespace RindowTest\OpenBLAS\FFI\LapackbTest;

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Buffer\FFI\Buffer;
use Rindow\OpenBLAS\FFI\Blas as OpenBLAS;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use TypeError;
use OutOfRangeException;
use ArrayObject;
use ArrayAccess;

require_once __DIR__.'/Utils.php';
use RindowTest\OpenBLAS\FFI\Utils;

class LapackbTest extends TestCase
{
    use Utils;

    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    public function getLapack()
    {
        $lapack = $this->factory->Lapackb();
        return $lapack;
    }

    public function translate_gesvd(NDArray $matrix,?bool $fullMatrices=null)
    {
        if($matrix->ndim()!=2) {
            throw new InvalidArgumentException("input array must be 2D array");
        }
        if($fullMatrices===null)
            $fullMatrices = true;
        [$m,$n] = $matrix->shape();
        $k = min($m, $n);
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $rowsU = $m; $colsU = $m;    // U is m x m
            $rowsVT = $n; $colsVT = $n; // VT is n x n
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $rowsU = $m; $colsU = $k;    // U is m x k
            //$rowsVT = $k; $colsVT = $n;// VT is k x n
            $rowsVT = $n; $colsVT = $n;  // bug int the lapacke
        }
        $ldA = $n;       // Input A is ROW_MAJOR (m x n), ld = n
        $ldU = $colsU;   // Output U is ROW_MAJOR (rowsU x colsU), ld = colsU
        $ldVT = $colsVT; // Output VT is ROW_MAJOR (rowsVT x colsVT), ld = colsVT = n

        $S = $this->zeros([$k],dtype:$matrix->dtype());
        // Allocate U and VT with correct ROW_MAJOR dimensions and leading dimensions
        $U = $this->zeros([$rowsU, $colsU],dtype:$matrix->dtype());
        $VT = $this->zeros([$rowsVT, $colsVT],dtype:$matrix->dtype());
        $SuperB = $this->zeros([$k-1],dtype:$matrix->dtype()); // Size is min(m,n)-1

        $AA = $matrix->buffer();
        $offsetA = $matrix->offset();
        $SS = $S->buffer();
        $offsetS = $S->offset();
        $UU = $U->buffer();
        $offsetU = $U->offset();
        $VVT = $VT->buffer();
        $offsetVT = $VT->offset();
        $SuperBB = $SuperB->buffer();
        $offsetSuperB = $SuperB->offset();
        return [
            self::LAPACK_ROW_MAJOR, // Specify layout
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,  // Pass ROW_MAJOR A and its ld
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,  // Pass ROW_MAJOR U buffer and its ld
            $VVT, $offsetVT, $ldVT, // Pass ROW_MAJOR VT buffer and its ld
            $SuperBB,  $offsetSuperB,

            $U,$S,$VT,$SuperB
        ];
        //if(!$fullMatrices) {
        //    // bug in the lapacke ???
        //    $VT = $this->copy($VT[[0,min($m,$n)-1]]);
        //}
        //return [$U,$S,$VT];
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

    #[DataProvider('providerDtypesFloats')]
    public function testSvdFull1($params)
    {
        extract($params);
        $lapack = $this->getLapack();
        $a = $this->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ],dtype:$dtype);
        //echo "---- a ----\n";
        //echo $this->arrayToString($a,'%10.6f',true)."\n";
        $fullMatrices = null;
        $this->assertEquals([6,5],$a->shape());
        [
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $u,$s,$vt,$superB
        ] = $this->translate_gesvd($a,$fullMatrices);

        $lapack->gesvd(
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );
        if(!$fullMatrices) {
            // bug in the lapacke ???
            //$vt = $this->copy($vt[[0,min($m,$n)-1]]);
        }
        // [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([6,6],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([5,5],$vt->shape());

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $this->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [-0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [-0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [-0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [-0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [ 0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ],dtype:$dtype);
        //$this->assertTrue(false);
        //echo "---- u ----\n";
        //echo $this->arrayToString($u,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($u,$correctU,rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
            ,dtype:$dtype);
        //echo "---- s ----\n";
        //echo $this->arrayToString($s,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($s,$correctS,rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $this->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ],dtype:$dtype);
        //echo "---- vt ----\n";
        //echo $this->arrayToString($vt,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($vt,$correctVT,rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($vt,$correctVT,-1))));
        # ---- superB ----
        //echo "---- superB ----\n";
        //echo $this->arrayToString($superB,'%10.6f',true)."\n";
    }

    #[DataProvider('providerDtypesFloats')]
    public function testSvdFull2($params)
    {
        extract($params);
        $lapack = $this->getLapack();
        $a = $this->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ],dtype:$dtype);
        $a = $this->transpose($a);
        //echo "input=".$this->arrayToString($a,'%10.6f',true)."\n";
        $this->assertEquals([5,6],$a->shape());
        $fullMatrices = null;
        [
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $u,$s,$vt,$superB
        ] = $this->translate_gesvd($a,$fullMatrices);
        //[$u,$s,$vt] = $la->svd($a);
        $lapack->gesvd(
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );
        $this->assertEquals([5,5],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([6,6],$vt->shape());

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $this->array([
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ],dtype:$dtype);
        $correctU = $this->transpose($correctU);
        $this->assertEquals([5,5],$correctU->shape());

        //echo "u=".$this->arrayToString($u,'%10.6f',true)."\n";
        //echo "correctU=".$this->arrayToString($correctU,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($this->absarray($u),$this->absarray($correctU),rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        ,dtype:$dtype);
        $this->assertTrue($this->isclose($this->absarray($s),$this->absarray($correctS),rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $this->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [ 0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [ 0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [-0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ],dtype:$dtype);
        $correctVT = $this->transpose($correctVT);
        $this->assertTrue($this->isclose($this->absarray($vt),$this->absarray($correctVT),rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue(true);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testSvdSmallU($params)
    {
        extract($params);
        $lapack = $this->getLapack();
        $a = $this->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ],dtype:$dtype);
        $fullMatrices = false;
        [
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $u,$s,$vt,$superB
        ] = $this->translate_gesvd($a,$fullMatrices);
        //[$u,$s,$vt] = $la->svd($a,$full_matrices=false);
        $lapack->gesvd(
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $this->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23],
            [-0.40, 0.24,-0.22,-0.75,-0.36],
            [-0.03,-0.60,-0.45, 0.23,-0.31],
            [-0.43, 0.24,-0.69, 0.33, 0.16],
            [-0.47,-0.35, 0.39, 0.16,-0.52],
            [ 0.29, 0.58,-0.02, 0.38,-0.65],
        ],dtype:$dtype);
        //echo $this->arrayToString($u,format:'%6.2f',indent:true)."\n";
        $this->assertTrue($this->isclose($this->absarray($u),$this->absarray($correctU),rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01],dtype:$dtype
        );
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        $this->assertTrue($this->isclose($this->absarray($s),$this->absarray($correctS),rtol:1e-2,atol:1e-3));
        # ---- vt ----
        $correctVT = $this->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ],dtype:$dtype);
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        $this->assertTrue($this->isclose($this->absarray($vt),$this->absarray($correctVT),rtol:1e-2,atol:1e-3));
        $this->assertTrue(true);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testSvdSmallVT($params)
    {
        extract($params);
        $lapack = $this->getLapack();
        $a = $this->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ],dtype:$dtype);
        $a = $this->transpose($a);
        $fullMatrices = false;
        [
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $u,$s,$vt,$superB
        ] = $this->translate_gesvd($a,$fullMatrices);
        //[$u,$s,$vt] = $la->svd($a,$full_matrices=false);
        $lapack->gesvd(
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #  echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $this->array([
            [ 0.25, 0.40, 0.69, 0.37, 0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ],dtype:$dtype);
        $correctU = $this->transpose($correctU);
        //$correctU = $this->square($correctU);
        //$u = $la->square($u);
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($u,$correctU,-1))));
        $this->assertTrue($this->isclose($this->absarray($u),$this->absarray($correctU),rtol:1e-2,atol:1e-3));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01],dtype:$dtype
        );
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($s,$correctS,-1))));
        $this->assertTrue($this->isclose($this->absarray($s),$this->absarray($correctS),rtol:1e-2,atol:1e-3));
        # ---- vt ----
        $correctVT = $this->array([
            [ 0.59, 0.26, 0.36, 0.31, 0.23,],
            [ 0.40, 0.24,-0.22,-0.75,-0.36,],
            [ 0.03,-0.60,-0.45, 0.23,-0.31,],
            [ 0.43, 0.24,-0.69, 0.33, 0.16,],
            [ 0.47,-0.35, 0.39, 0.16,-0.52,],
            [-0.29, 0.58,-0.02, 0.38,-0.65,],
        ],dtype:$dtype);
        $correctVT = $this->transpose($correctVT);
        $k = min($m, $n);
        if(!$fullMatrices) {
            // bug in the lapacke ???
            $vt = $this->array($vt->buffer(),$vt->dtype(),[$k,$n],0);
        }
        //$correctVT = $la->square($correctVT);
        //$vt = $la->square($vt);
        //$this->assertLessThan(0.01,abs($la->amax($la->axpy($vt,$correctVT,-1))));
        //echo $this->arrayToString($vt,format:'%6.2f',indent:true)."\n";
        $this->assertTrue($this->isclose($this->absarray($vt),$this->absarray($correctVT),rtol:1e-2,atol:1e-3));
        $this->assertTrue(true);
    }

}
