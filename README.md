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
Executables will appear in the bld directory of the respective application (e.g. `apps/mnist/bld/gcc/mnist.out`). In order to run on the device use mspdebug. Run,

1. `mspdebug -v 3300 -d /dev/ttyACM0 tilib`
2. \> `prog apps/mnist/bld/gcc/mnist.out`
3. \> `run`

### SONIC




