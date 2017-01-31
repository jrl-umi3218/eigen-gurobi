# eigen-gurobi

eigen-gurobi allow to use the Gurobi QP solver with the Eigen3 library.

## Installing

### Manual

#### Dependencies

To compile you need the following tools:

 * [Git]()
 * [CMake]() >= 2.8
 * [pkg-config]()
 * [doxygen]()
 * [g++]()
 * [Boost](http://www.boost.org/doc/libs/1_58_0/more/getting_started/unix-variants.html) >= 1.49
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2
 * [Gurobi](http://www.gurobi.com/index) >= 4.0

#### Building

```sh
git clone --recursive https://github.com/jrl-umi3218/eigen-gurobi
cd eigen-gurobi
mkdir _build
cd _build
cmake [options] ..
make && make intall
```

Where the main options are:

 * `-DCMAKE_BUIlD_TYPE=Release` Build in Release mode
 * `-DCMAKE_INSTALL_PREFIX=some/path/to/install` default is `/usr/local`
 * `-DUNIT_TESTS=ON` Build unit tests.
