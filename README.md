Rindow OpenBLAS PHP Interface
=============================
Status:
[![Build Status](https://github.com/rindow/rindow-openblas-ffi/workflows/tests/badge.svg)](https://github.com/rindow/rindow-openblas-ffi/actions)
[![Downloads](https://img.shields.io/packagist/dt/rindow/rindow-openblas-ffi)](https://packagist.org/packages/rindow/rindow-openblas-ffi)
[![Latest Stable Version](https://img.shields.io/packagist/v/rindow/rindow-openblas-ffi)](https://packagist.org/packages/rindow/rindow-openblas-ffi)
[![License](https://img.shields.io/packagist/l/rindow/rindow-openblas-ffi)](https://packagist.org/packages/rindow/rindow-openblas-ffi)

The Rindow OpenBLAS FFI is universal Buffer for N-dimension and OpenBLAS and Mathematical library.
It can be used via PHP's FFI interface.

- Provides Universal Buffer for 1-dimension for data exchange between C,C+ language and PHP.
- The OpenBLAS library available to PHP. Only the commonly used functions in OpenBLAS are provided.
- Provides commonly used Mathematical libraries not included in OpenBLAS.

You can do very fast N-dimensional array operations in conjunction with the [Rindow Math Matrix](https://github.com/rindow/rindow-math-matrix).

Very useful when you want to do deep learning with PHP!

Requirements
============

- PHP 8.1 or PHP8.2 or PHP8.3
- Linux or Windows
- OpenBLAS

How to download and setup
=========================

### Windows
The OpenBLAS Library release number is included in the filename of the rindow-openblas pre-built archive file.

- https://github.com/xianyi/OpenBLAS/releases

Unzip it to a suitable location and set the execution path in the bin directory.

```shell
TMP>set PATH=%PATH%;\path\to\OpenBLAS\bin
```

And then set it up using composer.

```shell
C> mkdir \your\app\dir
C> cd \your\app\dir
C> composer require rindow/rindow-openblas-ffi
```

### Ubuntu
Use the apt command to install the deb file. 

```shell
$ sudo apt install libopenblas-base liblapacke
```

And then set it up using composer.
```shell
$ mkdir \your\app\dir
$ cd \your\app\dir
$ composer require rindow/rindow-openblas-ffi
```
