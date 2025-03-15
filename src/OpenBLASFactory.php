<?php
namespace Rindow\OpenBLAS\FFI;

use FFI;
//use FFI\Env\Runtime as FFIEnvRuntime;
//use FFI\Env\Status as FFIEnvStatus;
//use FFI\Location\Locator as FFIEnvLocator;
use FFI\Exception as FFIException;
use RuntimeException;

class OpenBLASFactory
{
    private static ?FFI $ffi = null;
    private static ?FFI $ffiLapacke = null;
    private static ?FFI $ffiLapack = null;
    /** @var array<string,array<string,array<string,mixed>>> $configMatrix */
    protected array $configMatrix = [
        'WINNT' => [
            'blas' => [
                'header' => __DIR__.'/openblas.h',
                'libs' => ['libopenblas.dll'],
            ],  
            'lapacke' => [
                'header' => __DIR__ . '/lapacke.h',
                'libs' => ['libopenblas.dll'],
            ],
            'lapack' => [
                'header' => __DIR__ . '/lapack.h',
                'libs' => ['libopenblas.dll'],
            ],
        ],
        'Linux' => [
            'blas' => [
                'header' => __DIR__.'/openblas.h',
                'libs' => ['libopenblas.so.0'],
            ],
            'lapacke' => [
                'header' => __DIR__ . '/lapacke.h',
                'libs' => ['liblapacke.so.3'],
            ],
            'lapack' => [
                'header' => __DIR__ . '/lapack.h',
                //'libs' => ['/usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3'],
                //'libs' => ['/usr/lib/x86_64-linux-gnu/openblas-openmp/liblapack.so.3'],
                'libs' => ['liblapack.so.3'],
            ],
        ],
        'Darwin' => [
            'blas' => [
                'header' => __DIR__.'/cblas_new_vecLib.h',
                'libs' => ['/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib'],
            ],
            'lapacke' => [
                'header' => null,
                'libs' => null,
            ],
            'lapack' => [
                'header' => __DIR__ . '/clapack_vecLib.h',
                'libs' => ['/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib'],
            ],
        ],
    ];
    /** @var array<string> $errors */
    private array $errors = [];

    /**
     * @param array<string> $libFiles
     * @param array<string> $lapackeLibs
     */
    public function __construct(
        ?string $headerFile=null,
        ?array $libFiles=null,
        ?string $lapackeHeader=null,
        ?array $lapackeLibs=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        if(!extension_loaded('ffi')) {
            return;
        }

        $config = $this->generateConfig([
            'blas' => [
                'header' => $headerFile,
                'libs' => $libFiles,
            ],
            'lapacke' => [
                'header' => $lapackeHeader,
                'libs' => $lapackeLibs,
            ],
            'lapack' => [],
        ]);
        $drivers = $this->loadLibraries($config);
        if(isset($drivers['blas'])) {
            self::$ffi = $drivers['blas'];
        }
        if(isset($drivers['lapacke'])) {
            self::$ffiLapacke = $drivers['lapacke'];
        }
        if(isset($drivers['lapack'])) {
            self::$ffiLapack = $drivers['lapack'];
        }
    }

    /**
     * @return array<string>
     */
    public function errors() : array
    {
        return $this->errors;
    }

    /**
     * @param  array<string,array<string,mixed>> $params
     * @return array<string,array<string,mixed>>
     */
    protected function generateConfig(array $params) : array
    {
        $os = PHP_OS;
        if(!isset($this->configMatrix[$os])) {
            throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
        }
        $defaults = $this->configMatrix[$os];
        foreach($defaults as $type => $names) {
            foreach($names as $key => $value) {
                $params[$type][$key] ??= $value;
            }
        }
        return $params;
    }

    /**
     * @param array<array<mixed>> $params
     * @return array<mixed>
     */
    protected function loadLibraries(array $params) : array
    {
        $ffis = [];
        foreach($params as $key => $param) {
            if(!isset($param['header'])) {
                continue;
            }
            $code = file_get_contents($param['header']);
            if($code===false) {
                throw new RuntimeException('The header file not found: "'.$param['header'].'"');
            }
            foreach($param['libs'] as $filename) {
                $ffi = null;
                try {
                    $ffi = FFI::cdef($code,$filename);
                } catch(FFIException $e) {
                    $this->errors[] = $e->getMessage();
                    continue;
                }
                $ffis[$key] = $ffi;
                break;
            }
        }
        return $ffis;
    }

    public function isAvailable() : bool
    {
        return self::$ffi!==null;
        //$isAvailable = FFIEnvRuntime::isAvailable();
        //if(!$isAvailable) {
        //    return false;
        //}
        //$pathname = FFIEnvLocator::resolve(...$this->libs);
        //return $pathname!==null;
    }

    public function Blas() : Blas
    {
        if(self::$ffi==null) {
            throw new RuntimeException('openblas library not loaded.');
        }
        return new Blas(self::$ffi);
    }

    public function Lapack() : Lapack
    {
        if(PHP_OS=='Darwin') {
            return $this->Lapackb();
        }
        if(self::$ffiLapacke==null) {
            throw new RuntimeException('lapacke library not loaded.');
        }
        return new Lapacke(self::$ffiLapacke);
    }

    public function Lapackb() : Lapack
    {
        if(self::$ffiLapack==null) {
            throw new RuntimeException('lapack library not loaded.');
        }
        return new Lapackb(self::$ffiLapack, self::$ffi);
    }
}
