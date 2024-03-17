<?php
namespace RindowTest\OpenBLAS\FFI\ReleaseTest;

use PHPUnit\Framework\TestCase;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use Rindow\OpenBLAS\FFI\Blas;
use Rindow\OpenBLAS\FFI\Lapack;
use FFI;

class ReleaseTest extends TestCase
{
    public function testFFINotLoaded()
    {
        $factory = new OpenBLASFactory();
        if(extension_loaded('ffi')) {
            $blas = $factory->Blas();
            $lapack = $factory->Lapack();
            $this->assertInstanceof(Blas::class,$blas);
            $this->assertInstanceof(Lapack::class,$lapack);
        } else {
            $this->assertFalse($factory->isAvailable());
        }
    }
}