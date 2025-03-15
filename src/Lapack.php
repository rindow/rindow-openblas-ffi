<?php
namespace Rindow\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\LinearBuffer as BufferInterface;

interface Lapack
{
    public function ffi() : object;

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
    ) : void;

}