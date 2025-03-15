<?php
namespace RindowTest\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Interop\Polite\Math\Matrix\Buffer as BufferInterface;
use ArrayObject;
use InvalidArgumentException;
use Rindow\Math\Buffer\FFI\Buffer;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;

function C(
    ?float $r=null,
    ?float $i=null,
) : object
{
    $r = $r ?? 0.0;
    $i = $i ?? 0.0;
    return (object)['real'=>$r,'imag'=>$i];
}

trait Utils
{
    protected $factory;
    
    public function setUp() : void
    {
        $this->factory = new OpenBLASFactory();
    }

    public function getBlas()
    {
        $blas = $this->factory->Blas();
        return $blas;
    }

    public function translate_scal(
        float|object $a,NDArray $X) : array
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return [$N,$a,$XX,$offX,1];
    }

    public function translate_copy(
        NDArray $X,
        NDArray $Y ) : array
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
        return [$N,$XX,$offX,1,$YY,$offY,1];
    }

    public function translate_axpy(
        NDArray $X,NDArray $Y, float|object|null $alpha=null) : array
    {
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($alpha===null) {
            if($this->isComplex($X->dtype())) {
                $alpha = C(1.0);
            } else {
                $alpha = 1.0;
            }
        }
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $N = $X->size();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        return [$N,$alpha,$XX,$offX,1,$YY,$offY,1];
    }

    public function translate_amin(
        NDArray $X) : array
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return [$N,$XX,$offX,1];
    }

    protected function fp64() : bool
    {
        return true;
    }

    public function array(mixed $array=null, ?int $dtype=null, ?array $shape=null) : object
    {
        $ndarray = new class ($array, $dtype, $shape) implements NDArray {
            protected object $buffer;
            protected int $size;
            protected int $dtype;
            protected int $offset;
            protected array $shape;
            public function __construct(mixed $array=null, ?int $dtype=null, ?array $shape=null,?int $offset=null) {
                $dtype = $dtype ?? NDArray::float32;
                if($array===null && $shape!==null) {
                    $this->checkShape($shape);
                    $size = (int)array_product($shape);
                    $buffer = $this->newBuffer($size,$dtype);
                    $offset = 0;
                } elseif($array instanceof BufferInterface) {
                    if($offset===null||!is_int($offset)) {
                        throw new InvalidArgumentException("Must specify offset with the buffer");
                    }
                    if($shape===null) {
                        throw new InvalidArgumentException("Invalid dimension size");
                    }
                    $this->buffer = $array;
                    $this->offset = $offset;
                    $buffer = $array;
                } elseif(is_array($array)||$array instanceof ArrayObject) {
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

            function newBuffer($size,$dtype) : object
            {
                return new Buffer($size,$dtype);
            }
            
            protected function isComplex(?int $dtype=null) : bool
            {
                $dtype = $dtype ?? $this->_dtype;
                return $dtype==NDArray::complex64||$dtype==NDArray::complex128;
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
                    if($this->isComplex($this->dtype)) {
                        for($i=0; $i<$size; $i++) {
                            $v = $F[$idx];
                            $A[$i] = C($v->real,$v->imag);
                            $idx++;
                        }
                    } else {
                        for($i=0; $i<$size; $i++) {
                            $A[$i] = $F[$idx];
                            $idx++;
                        }
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
                    $v = $this->buffer[$this->offset];
                    if($this->isComplex($this->dtype)) {
                        $v = C($v->real,$v->imag);
                    }
                    return $v;
                }
                $idx = $this->offset;
                return $this->flat2Array($this->buffer, $idx, $this->shape);
            }

            public function shape() : array { return $this->shape; }

            public function ndim() : int { return count($this->shape); }
        
            public function dtype() { return $this->dtype; }
        
            public function buffer() : BufferInterface { return $this->buffer; }
        
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
                    $v = $this->buffer[$this->offset+$offset];
                    if($this->isComplex($this->dtype)) {
                        $v = C($v->real,$v->imag);
                    }
                    return $v;
                }
                $size = (int)array_product($shape);
                $new = new self($this->buffer,$this->dtype,$shape,$this->offset+$offset*$size);
                return $new;
            }
        
            public function offsetSet( $offset , $value ) : void { throw new \Excpetion('not implement'); }
            public function offsetUnset( $offset ) : void { throw new \Excpetion('not implement'); }
            public function count() : int  { throw new \Excpetion('not implement'); }
            public function  getIterator() : Traversable  { throw new \Excpetion('not implement'); }
        };
        return $ndarray;
    }

    public function alloc(array $shape,?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        return $ndarray;
    }

    public function zeros(array $shape,?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        return $ndarray;
    }

    public function ones(array $shape, ?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        $buffer = $ndarray->buffer();
        $dtype = $ndarray->dtype();
        $value = 1;
        if($this->isComplex($dtype)) {
            $value = $this->toComplex($value);
        }
        $size = count($buffer);
        for($i=0;$i<$size;++$i) {
            $buffer[$i] = $value;
        }
        return $ndarray;
    }

    public function full(array $shape, mixed $value ,?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        $buffer = $ndarray->buffer();
        $size = count($buffer);
        for($i=0;$i<$size;++$i) {
            $buffer[$i] = $value;
        }
        return $ndarray;
    }

    protected function isComplex($dtype) : bool
    {
        return $dtype==NDArray::complex64||$dtype==NDArray::complex128;
    }

    protected function toComplex(mixed $array) : mixed
    {
        if(!is_array($array)) {
            if(is_numeric($array)) {
                return C($array,i:0);
            } else {
                return C($array->real,i:$array->imag);
            }
        }
        $cArray = [];
        foreach($array as $value) {
            $cArray[] = $this->toComplex($value);
        }
        return $cArray;
    }

    protected function complementTrans(?bool $trans,?bool $conj,int $dtype) : array
    {
        $trans = $trans ?? false;
        if($this->isComplex($dtype)) {
            $conj = $conj ?? $trans;
        } else {
            $conj = $conj ?? false;
        }
        return [$trans,$conj];
    }

    protected function transToCode(bool $trans,bool $conj) : int
    {
        if($trans) {
            return $conj ? BLAS::ConjTrans : BLAS::Trans;
        } else {
            return $conj ? BLAS::ConjNoTrans : BLAS::NoTrans;
        }
    }

    protected function abs(float|int|object $value) : float
    {
        if(is_numeric($value)) {
            return abs($value);
        }
        $abs = sqrt(($value->real)**2+($value->imag)**2);
        return $abs;
    }

    protected function copy(NDArray $x,?NDArray $y=null) : NDArray
    {
        $blas = $this->getBlas();

        if($y==null) {
            $y = $this->zeros($x->shape(),dtype:$x->dtype());
        }
        $blas->copy(...$this->translate_copy($x,$y));
        return $y;
    }

    protected function isclose(NDArray $a, NDArray $b, ?float $rtol=null, ?float $atol=null) : bool
    {
        $blas = $this->getBlas();

        $isCpx = $this->isComplex($a->dtype());
        if($rtol===null) {
            $rtol = $isCpx?C(1e-04):1e-04;
        }
        if($atol===null) {
            $atol = 1e-07;
        }
        if($a->shape()!=$b->shape()) {
            return false;
        }
        // diff = b - a
        $alpha =  $isCpx?C(-1):-1;
        $diffs = $this->copy($b);
        $blas->axpy(...$this->translate_axpy($a,$diffs,$alpha));
        $iDiffMax = $blas->iamax(...$this->translate_amin($diffs));
        $diff = $this->abs($diffs->buffer()[$iDiffMax]);

        // close = atol + rtol * b
        $scalB = $this->copy($b);
        $blas->scal(...$this->translate_scal($rtol,$scalB));
        $iCloseMax = $blas->iamax(...$this->translate_amin($scalB));
        $close = $atol+$this->abs($scalB->buffer()[$iCloseMax]);
        return $diff < $close;
    }

    public function arrayToString(
        NDArray $array,
        ?string $format=null,
        bool|int|null $indent=null) : string
    {
        $shape = $array->shape();
        if(count($shape)==0) {
            $value = $array->toArray();
            if($format) {
                return sprintf($format,$value);
            } else {
                return strval($value);
            }
        }
        $n = array_shift($shape);
        if(!is_numeric($indent) && $indent===true) {
            $indent=1;
        }
        if(count($shape)==0) {
            if($array->dtype()==NDArray::bool) {
                $str = '';
                foreach($array->toArray() as $value) {
                    $str .= ($str==='') ? '[' : ',';
                    $str .= $value ? 'true' : 'false';
                }
                $str .= ']';
                return $str;
            } else {
                if($format) {
                    return '['.implode(',',array_map(function($x) use ($format,$array) {
                            if($array->dtype()==NDArray::complex64||$array->dtype()==NDArray::complex128) {
                                return sprintf($format,$x->real,$x->imag);
                            } else {
                                return sprintf($format,$x);
                            }
                        },$array->toArray())).']';
                } else {
                    return '['.implode(',',$array->toArray()).']';
                }
            }
        }
        $string = '[';
        if($indent) {
            $string .= "\n";
        }
        for($i=0;$i<$n;$i++) {
            if($i!=0) {
                $string .= ',';
                if($indent) {
                    $string .= "\n";
                }
            }
            if($indent) {
                $string .= str_repeat(' ',$indent);
                $string .= $this->arrayToString($array[$i],$format,$indent+1);
            } else {
                $string .= $this->arrayToString($array[$i],$format,$indent);
            }
        }
        if($indent) {
            $string .= "\n";
            $string .= str_repeat(' ',$indent-1);
        }
        $string .= ']';
        return $string;
    }

}