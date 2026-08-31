<?php
define('COMPOSER_HOME', getenv('COMPOSER_HOME'));
if(COMPOSER_HOME && file_exists(COMPOSER_HOME.'/vendor/autoload.php')) {
    $loader = include COMPOSER_HOME.'/vendor/autoload.php';
} else {
    throw new \Exception("Loader is not found.");
}
$loader->addPsr4('Rindow\\OpenBLAS\\FFI\\',__DIR__.'/../src');
$loader->addPsr4('Rindow\\Math\\Buffer\\FFI\\',__DIR__.'/../../rindow-math-buffer-ffi/src');
$loader->addPsr4('Interop\\Polite\\Math\\', __DIR__.'/../../../interop-phpobjects/polite-math/src');

return $loader;
