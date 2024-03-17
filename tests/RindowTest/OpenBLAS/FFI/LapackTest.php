<?php
namespace RindowTest\OpenBLAS\FFI\LapackTest;

use PHPUnit\Framework\TestCase;
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

class LapackTest extends TestCase
{
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected $factory;
    protected $blas;

    public function setUp() : void
    {
        $this->factory = new OpenBLASFactory();
        $this->blas = $this->factory->Blas();
    }

    public function getLapack()
    {
        $lapack = $this->factory->Lapack();
        return $lapack;
    }

    public function zeros(array $shape,int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        return $ndarray;
    }

    public function array(array $array=null, int $dtype=null, array $shape=null) : object
    {
        $ndarray = new class ($array, $dtype, $shape) implements NDArray {
            protected object $buffer;
            protected int $size;
            protected int $dtype;
            protected int $offset;
            protected array $shape;
            public function __construct(
                array|BufferInterface $array=null, int $dtype=null, array $shape=null,int $offset=null,
            )
            {
                $dtype = $dtype ?? NDArray::float32;
                if(is_array($array)||$array instanceof ArrayObject) {
                    $dummyBuffer = new ArrayObject();
                    $idx = 0;
                    $this->array2Flat($array,$dummyBuffer,$idx,$prepare=true);
                    $buffer = $this->newBuffer($idx,$dtype);
                    $idx = 0;
                    $this->array2Flat($array,$buffer,$idx,$prepare=false);
                    $offset = 0;
                    if($shape===null) {
                        $shape = $this->genShape($array);
                    }
                } elseif(is_numeric($array)||is_bool($array)) {
                    if(is_bool($array)&&$dtype!=NDArray::bool) {
                        throw new InvalidArgumentException("unmatch dtype with bool value");
                    }
                    $buffer = $this->newBuffer(1,$dtype);
                    $buffer[0] = $array;
                    $offset = 0;
                    if($shape===null) {
                        $shape = [];
                    }
                    $this->checkShape($shape);
                    if(array_product($shape)!=1)
                        throw new InvalidArgumentException("Invalid dimension size");
                } elseif($array===null && $shape!==null) {
                    $this->checkShape($shape);
                    $size = (int)array_product($shape);
                    $buffer = $this->newBuffer($size,$dtype);
                    $offset = 0;
                } elseif($this->isBuffer($array)) {
                    if($offset===null||!is_int($offset))
                        throw new InvalidArgumentException("Must specify offset with the buffer");
                    if($shape===null)
                        throw new InvalidArgumentException("Invalid dimension size");
                    $buffer = $array;
                    $offset = $offset;
                } else {
                    var_dump($array);var_dump($shape);
                    throw new \Exception("Illegal array type");
                }
                $this->buffer = $buffer;
                $this->size = $buffer->count();
                $this->dtype = $buffer->dtype();
                $this->shape = $shape;
                $this->offset = $offset;
            }

            protected function newBuffer($size,$dtype) : object
            {
                return new Buffer($size,$dtype);
            }
            
            protected function isBuffer($buffer)
            {
                if($buffer instanceof BufferInterface) {
                    return true;
                } else {
                    return false;
                }
            }
        
            protected function array2Flat($A, $F, &$idx, $prepare)
            {
                if(is_array($A)) {
                    ksort($A);
                } elseif($A instanceof ArrayObject) {
                    $A->ksort();
                }
        
                $num = null;
                foreach ($A as $key => $value) {
                    if(!is_int($key))
                        throw new InvalidArgumentException("Dimension must be integer");
                    if(is_array($value)||$value instanceof ArrayObject) {
                        $num2 = $this->array2Flat($value, $F, $idx, $prepare);
                        if($num===null) {
                            $num = $num2;
                        } else {
                            if($num!=$num2)
                                throw new InvalidArgumentException("The shape of the dimension is broken");
                        }
                    } else {
                        if($num!==null)
                            throw new InvalidArgumentException("The shape of the dimension is broken");
                        if(!$prepare)
                            $F[$idx] = $value;
                        $idx++;
                    }
                }
                return count($A);
            }

            protected function flat2Array($F, &$idx, array $shape)
            {
                $size = array_shift($shape);
                if(count($shape)) {
                    $A = [];
                    for($i=0; $i<$size; $i++) {
                        $A[$i] = $this->flat2Array($F,$idx,$shape);
                    }
                }  else {
                    $A = [];
                    for($i=0; $i<$size; $i++) {
                        $A[$i] = $F[$idx];
                        $idx++;
                    }
                }
                return $A;
            }
                
            protected function genShape($A)
            {
                $shape = [];
                while(is_array($A) || $A instanceof ArrayObject) {
                    $shape[] = count($A);
                    $A = $A[0];
                }
                return $shape;
            }
        
            protected function checkShape(array $shape)
            {
                foreach($shape as $num) {
                    if(!is_int($num)) {
                        throw new InvalidArgumentException(
                            "Invalid shape numbers. It gives ".gettype($num));
                    }
                    if($num<=0) {
                        throw new InvalidArgumentException(
                            "Invalid shape numbers. It gives ".$num);
                    }
                }
            }

            public function toArray()
            {
                if(count($this->shape)==0) {
                    return $this->buffer[$this->offset];
                }
                $idx = $this->offset;
                return $this->flat2Array($this->buffer, $idx, $this->shape);
            }

            public function shape() : array { return $this->shape; }

            public function ndim() : int { return count($this->shape); }
        
            public function dtype() { return $this->dtype; }
        
            public function buffer() : ArrayAccess { return $this->buffer; }
        
            public function offset() : int { return $this->offset; }
        
            public function size() : int { return $this->buffer->count(); }
        
            public function reshape(array $shape) : NDArray
            {
                if(array_product($shape)==array_product($this->shape)) {
                    $this->shape = $shape;
                } else {
                    throw new \Exception("unmatch shape");
                }
                return $this;
            }

            public function offsetExists( $offset ) : bool
            {
                if(count($this->shape)==0)
                    return false;
                if(is_array($offset)) {
                    if(count($offset)!=2 ||
                        !array_key_exists(0,$offset) || !array_key_exists(1,$offset) ||
                        $offset[0]>$offset[1]) {
                            $det = '';
                            if(is_numeric($offset[0])&&is_numeric($offset[1]))
                                $det = ':['. implode (',',$offset).']';
                            throw new OutOfRangeException("Illegal range specification.".$det);
                    }
                    $start = $offset[0];
                    $end   = $offset[1];
                } elseif(is_int($offset)) {
                    $start = $offset;
                    $end   = $offset;
                } else {
                    throw new OutOfRangeException("Dimension must be integer");
                }
                if($start < 0 || $end >= $this->shape[0])
                    return false;
                return true;
            }
        
            public function offsetGet( $offset ) : mixed
            {
                if(!$this->offsetExists($offset))
                    throw new OutOfRangeException("Index is out of range");
        
                // for range spesification
                if(is_array($offset)) {
                    $shape = $this->shape;
                    array_shift($shape);
                    $rowsCount = $offset[1]-$offset[0]+1;
                    if(count($shape)>0) {
                        $itemSize = (int)array_product($shape);
                    } else {
                        $itemSize = 1;
                    }
                    if($rowsCount<0) {
                        throw new OutOfRangeException('Invalid range');
                    }
                    array_unshift($shape,$rowsCount);
                    $size = (int)array_product($shape);
                    $new = new self($this->buffer,$this->dtype,$shape,$this->offset+$offset[0]*$itemSize);
                    return $new;
                }
        
                // for single index specification
                $shape = $this->shape;
                $max = array_shift($shape);
                if(count($shape)==0) {
                    return $this->buffer[$this->offset+$offset];
                }
                $size = (int)array_product($shape);
                $new = new self($this->buffer,$this->dtype,$shape,$this->offset+$offset*$size);
                return $new;
            }
            public function offsetSet( $offset , $value ) : void { throw new LogicException('not implement'); }
            public function offsetUnset( $offset ) : void { throw new LogicException('not implement'); }
            public function count() : int  { throw new LogicException('not implement'); }
            public function  getIterator() : Traversable  { throw new LogicException('not implement'); }
        };
        return $ndarray;
    }

    public function translate_gesvd(NDArray $matrix,$fullMatrices=null)
    {
        if($matrix->ndim()!=2) {
            throw new InvalidArgumentException("input array must be 2D array");
        }
        if($fullMatrices===null)
            $fullMatrices = true;
        [$m,$n] = $matrix->shape();
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $ldA = $n;
            $ldU = $m;
            $ldVT = $n;
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $ldA = $n;
            $ldU = min($m,$n);
            #$ldVT = min($m,$n);
            $ldVT = $n; // bug in the lapacke ???
        }

        $S = $this->zeros([min($m,$n)],$matrix->dtype());
        $U = $this->zeros([$m,$ldU],$matrix->dtype());
        $VT = $this->zeros([$ldVT,$n],$matrix->dtype());
        $SuperB = $this->zeros([min($m,$n)-1],$matrix->dtype());

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
            self::LAPACK_ROW_MAJOR,
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $U,$S,$VT
        ];
        //if(!$fullMatrices) {
        //    // bug in the lapacke ???
        //    $VT = $this->copy($VT[[0,min($m,$n)-1]]);
        //}
        //return [$U,$S,$VT];
    }

    public function amax(
        NDArray $X) : float
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $i = $this->blas->iamax($N,$XX,$offX,1);
        return $XX[$offX+$i];
    }

    public function axpy(
        NDArray $X,
        NDArray $Y,
        float $alpha=null) : NDArray
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
        if($alpha===null) {
            $alpha = 1.0;
        }
        $this->blas->axpy($N,$alpha,$XX,$offX,1,$YY,$offY,1);
        return $Y;
    }

    public function copy(
        NDArray $X,
        NDArray $Y=null ) : NDArray
    {
        if($Y===null) {
            $Y = $this->zeros($X->shape(),$X->dtype());
        } else {
            if($X->shape()!=$Y->shape()) {
                $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $this->blas->copy($N,$XX,$offX,1,$YY,$offY,1);
        return $Y;
    }

    public function testSvdFull1()
    {
        $lapack = $this->getLapack();
        $a = $this->array([
            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
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

            $u,$s,$vt
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
            $vt = $this->copy($vt[[0,min($m,$n)-1]]);
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
        ]);
        //$this->assertTrue(false);
        $this->assertLessThan(0.01,abs($this->amax($this->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        $this->assertLessThan(0.01,abs($this->amax($this->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $this->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        $this->assertLessThan(0.01,abs($this->amax($this->axpy($vt,$correctVT,-1))));
    }

}
