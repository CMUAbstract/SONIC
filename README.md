# SONIC & TAILS

### Overview
SONIC & TAILS are two runtime systems for doing inference on intermittent embedded devices. SONIC is software only, while TAILS relies on hardware acceleration available on MSP430 devices with support for the Low-energy Accelerator (LEA). Both systems are built upon a task-based model (Alpaca) and target the MSP430 platform. This repository is a top-level repository, containing two example applications that utilize SONIC & TAILS.

Please cite/refer to this [paper]() for more information.

### Downloading
This repository is a top-level repository that relies on git submodules for dependencies; when cloning please do the following:

`git clone --recursive https://github.com/CMUAbstract/SONIC`

### Directory/File Structure
We utilize a custom build system that builds all library dependencies as well as example applications. Below is an explanation of the directories/files found in this repo.

```
app/
    mnist/
        src/
            headers/
                conv1.h
                conv2.h
                fc1.h
                fc2.h
        main.c
    test/
        ...
ext/
    libfixed/
    libdnn/
    libmat/
    ...
params/
    mnist/
        conv1_md.param
        ...
    test/
        ...
scripts/
    gen_headers.py
    input.py
    int_test.py
    tf_test.py
    unit_test.py
tools/
    maker/
        ...
```
`app/` contains the `mnist/` and `test/` example applications. MNIST is an implementation of LeNet while test contains a series of unit tests. Source files can be found in their respective `src/` directories. Weights can be found in their respective `src/headers/` directories.

`ext/` contains libraries. `libdnn`, `libfixed`, and `libmat` are the relevant libraries for SONIC and TAILS (the rest provide additional functionality required to run on MSP430; i.e. console printing, access to pins, etc.). `libdnn` contains source code for the linear algebra and neural network operations for both SONIC & TAILS. `libfixed` is a fixed point math library. `libmat` is a matrix data structure library.

`params` contains raw parameters from tensorflow (in the case of MNIST) and for the unit tests.

`scripts` contains several helpful python scripts for testing. `int_test.py`, `tf_test.py`, and `unit_test.py` are testing scripts for comparison with output from MSP430. `int_test.py` and `tf_test.py` are for MNIST (both are full precision, but `int_test.py` can be made to reflect fixed point arithmetic). `unit_test.py` is for the test application and prints out a series of unit test results. `gen_headers.py` can be used to generate headers from dumps of raw parameters from Tensorflow. Please refer to the main function in that file for an example of how to utilize it.

`tools` contains the maker build system.

### Building
The following list of commands summarizes how to build the two example applications and how to change build parameters in order to utilize TAILS or SONIC for various contexts. Replace <APP> with the target application, so for mnist `<APP>` is mnist and for test `<APP>` is test.

1. Clean dependencies: `make apps/<APP>/bld/gcc/depclean`
2. Build dependencies: `make apps/<APP>/bld/gcc/depclean`
3. Build target: `make apps/<APP>/bld/gcc/all BACKEND=sonic`
    - `BACKEND` determines which backend to use. Set to sonic for SONIC, tails for TAILS.
    - `CONSOLE` set to one to enable printf debugging.
    - `CONT` set to one if running on continuous power
    - `INTERMITTENT` set to one if running on intermittent power.
    - **NOTE:** when changing build arguments remember to run commands 1 and 2. Since build arguments only change what is valid code in the files, make will not see any changes to dependencies, so a full clean must be done in order for the build arguments to take effect.

### Flashing
Executables will appear in the `bld` directory of the respective application (e.g. `apps/mnist/bld/gcc/mnist.out`). In order to run on the device use mspdebug. Run,

1. `mspdebug -v 3300 -d /dev/ttyACM0 tilib`
2. \> `prog apps/mnist/bld/gcc/mnist.out`
3. \> `run`

### Using SONIC or TAILS
Please refer to `main.c` for the mnist example application for a full example of how to use SONIC and TAILS. The following briefly describes how to construct an application that utilizes SONIC or TAILS.

SONIC and TAILS provide a unified interface with a set of common linear algebra and neural-network operations. Specifically the following operations are supported:

1. Nonlinear: Relu, Norm, MaxPool
2. Dense Scalar: addition, multiplication, division
3. Dense Matrix: matrix-vector multiplication, matrix-matrix addition, matrix-matrix 3D convolution.
4. Sparse Matrix: matrix-vector multiplication, matrix-matrix addition, matrix-matrix 3D convolution.
5. Other: dense zeroing of a matrix

The above functions operate on matrices and utilized fixed-point (Q10.5) math. The matrix data structure contains matrix data and metadata.
```
__ro_fram mat_t  mat_conv2_w = { // __ro_fram is a macro that places matrix in nvm
  .dims = {CONV2_WM_LEN}, // dense dimensions
  .len_dims = 1, // dimensionality (i.e. 3 for 3D, 1 if sparse)
  .strides = {1}, // offsets between dimensions (i.e. if dimension are 100x5x5 strides are 25, 5, 1)
  .data = conv2_wm, // data
  .sparse = { // sparse metadata
    .dims = {100, 20, 5, 5}, // sparse dimensions
    .len_dims = 4, // dimensionality
    .sizes = conv2_wm_sizes, // filter sizes (count of nonzero elements in filters)
    .offsets = conv2_wm_offsets, // column offsets
  }
};
```
To call a particular operation do the following:
```
/* 
  Assumes b, w, dest, src in that order
  b_ptr is the biases
  w_ptr is the weights
  b1 destination matrix
  b2 source matrix 
*/
PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);

// Set a task to return to after the operation is completed
TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
TRANSITION_TO(task_s_conv);
```
SONIC and TAILS uses a stack to pass arguments. Please pass arguments in the order specified by the comments. Not all operations take biases or weights and can be omitted from `PUSH_STACK` in those cases.

Some operations like convolution take other parameters. These parameters include x stride, y stride, and potentially x size and y size (used for maxpooling). To pass these parameters modify the global parameter data structure.

```
  params.same_padding = false; // same_padding
  params.size[0] = 1;
  params.size[1] = 2; // x size
  params.size[2] = 2; // y size
  params.stride[0] = 1;
  params.stride[1] = 1; // x stride
  params.stride[2] = 1; // y stride
```

#### Helpful Matrix Operations
* `MAT_RESHAPE` reshapes a matrix. Pass matrix (pointer) and series of dimensions to reshape matrix to.
* `MAT_DUMP` dumps a slice of 3D matrix. Pass matrix (pointer) and slice index.

#### Helpful Fixed-point Operations
* `F_LIT(2.34)` convert a floating-point number to fixed-point
* `F_TO_FLOAT(345)` convert a fixed-point number to floating-point
* **NOTE:** see fixed.h in libfixed for a list of all other fixed-point operations supported. Also see Makefile.options in libfixed for more configuration operations of the library.

### libdnn
libdnn contains the source of both SONIC and TAILS. The seires of Makefiles handle different options for each backend and also determine the backend being built. The `src` folder contains shared source files as well as `sonic`, `tails`, and `include` directories. `include` contains the header files required for linking. `sonic` contains the SONIC specific files and `tails` contains the same for the TAILS backend.

* buffer.c - unified storage buffers
* cleanup.c - cleans up task variables that need to be reset on transition
* linalg.c - contains a norm function
* misc.c
* nn.c – contains neural network operations such as fully-connected layer and convolution layer
* state.c – handles arguments passing to libdnn

Both `sonic/` and `tails/` contain the following files:

* nonlinear.c – contains relu function
* task\_dm\_conv.c – dense matrix-matrix convolution
* task\_ds\_add.c – dense scalar addition
* task\_ds\_mul.c – dense scalar multiplication
* task\_sm\_conv.c – sparse-dense matrix-matrix convolution (weights sparse, activations dense)
* task\_svm\_mul.c – sparse matrix-vector multiplication (weights sparse, activation vector dense)
* task\_dm\_add.c – dense matrix addition
* task\_dm\_mul.c – dense matrix-vector multiplication
* task\_ds\_div.c – dense scalar division
* task\_ds\_zero.c – dense zeroing function (sets all values in matrix to zero)
* task\_sm\_mul.c – sparse matrix-matrix multiplication (partially implemented)

Some of the above implementations are shared between SONIC and TAILS because sometimes it was impossible to efficiently leverage hardware acceleration.




