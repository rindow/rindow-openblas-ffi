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
- Windows or Linux(Ubuntu 20.04 or Debian 12 or later)
- OpenBLAS

How to download and setup
=========================

### Windows
The OpenBLAS Library release number is included in the filename of the rindow-openblas pre-built archive file.

- https://github.com/OpenMathLib/OpenBLAS/releases

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

### Linux
Install openblas with apt command
```shell
$ sudo apt install libopenblas0-openmp liblapacke
```

And then set it up using composer.
```shell
$ mkdir \your\app\dir
$ cd \your\app\dir
$ composer require rindow/rindow-openblas-ffi
```


### Troubleshooting for Linux
Since rindow-matlib currently uses OpenMP, choose the OpenMP version for OpenBLAS as well.

Using the pthread version of OpenBLAS can cause conflicts and become unstable and slow.
This issue does not occur on Windows.

If you have already installed the pthread version of OpenBLAS,
```shell
$ sudo apt remove libopenblas0-pthread
```

But if you can't remove it, you can switch to it using the update-alternatives command.

```shell
$ sudo update-alternatives --config libopenblas.so.0-x86_64-linux-gnu
$ sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu
```

If you really want to use the pthread version of OpenBLAS, please switch to the serial version of rindow-matlib.

There are no operational mode conflicts with OpenBLAS on Windows.

But, If you really want to use the pthread version of OpenBLAS, please switch to the serial version of rindow-matlib.

```shell
$ sudo update-alternatives --config librindowmatlib.so
There are 2 choices for the alternative librindowmatlib.so (providing /usr/lib/librindowmatlib.so).

  Selection    Path                                             Priority   Status
------------------------------------------------------------
* 0            /usr/lib/rindowmatlib-openmp/librindowmatlib.so   95        auto mode
  1            /usr/lib/rindowmatlib-openmp/librindowmatlib.so   95        manual mode
  2            /usr/lib/rindowmatlib-serial/librindowmatlib.so   90        manual mode

Press <enter> to keep the current choice[*], or type selection number: 2
```
Choose the "rindowmatlib-serial".

