## conda env

> -- Found libdw: /home/kerr1gan/anaconda3/envs/infini-venv/lib/libdw.so\
> -- Found libdwarf: /home/kerr1gan/anaconda3/envs/infini-venv/lib/libdwarf.so\

```bash
conda install -c conda-forge libdwarf
conda install -c conda-forge elfutils
```

## cmakelist.txt

specify the path of gcc, g++ and the include_dir manually
```txt
set(CMAKE_C_COMPILER "/home/kerr1gan/anaconda3/envs/infini-venv/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/kerr1gan/anaconda3/envs/infini-venv/bin/g++")

include_directories("/home/kerr1gan/anaconda3/envs/infini-venv/include")
```

specify the path of cuDNN manually
```txt
if(USE_CUDA)
  add_compile_definitions(USE_CUDA=1)
  # Since enable_language only executes once, rerun cmake is required if CMAKE_CUDA_HOST_COMPILER is wrong
  set(CMAKE_CUDA_HOST_COMPILER
      ${CMAKE_CXX_COMPILER}
      CACHE STRING "Set cuda host compiler path")
  # CMP0104 requires CUDA_ARCHITECTURES
  set_target_properties(InfiniTensor PROPERTIES CUDA_ARCHITECTURES "70;80")
  enable_language(CUDA)
  find_package(CUDAToolkit) # For nvrtc and cuda driver
  # fix error: cannot find ludnn
  set(CUDNN_INCLUDE_DIR /usr/local/cuda-11.8/include)
  set(CUDNN_LIBRARY /usr/local/cuda-11.8/lib64/libcudnn.so.8)
  # target_link_libraries(InfiniTensor cudnn CUDA::curand CUDA::cublas CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
  target_link_libraries(InfiniTensor ${CUDNN_LIBRARY} CUDA::curand CUDA::cublas CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
  if (BUILD_DIST)
    message(STATUS "Add BUILD_DIST, use NCCL with CUDA")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(NCCL REQUIRED)
    add_compile_definitions(INFINI_USE_NCCL=1)
    target_link_libraries(InfiniTensor nccl)
  endif()
endif()

```
## some else

activate the infini-venv manually, don't specify the env in a python file in vscode to activate the env automatically

that will result in error while make

```bash
/home/kerr1gan/.infini/include/handle/../export.h:13:13: error: expected ',' or '...' before 'extern'
   13 | #define __C extern "C"
      |             ^~~~~~
/home/kerr1gan/anaconda3/envs/normal-venv/lib/gcc/x86_64-conda-linux-gnu/11.4.0/include/emmintrin.h: In function 'void _mm_maskmoveu_si128(__m128i, __m128i, char*)':
/home/kerr1gan/.infini/include/handle/../export.h:13:13: error: expected primary-expression before 'extern'
   13 | #define __C extern "C"
      |             ^~~~~~
[ 17%] Building CXX object CMakeFiles/InfiniTensor.dir/src/core/tensor.cc.o
[ 18%] Building CXX object CMakeFiles/InfiniTensor.dir/src/core/tensor_base.cc.o
make[3]: *** [CMakeFiles/InfiniTensor.dir/build.make:233: CMakeFiles/InfiniTensor.dir/src/core/runtime.cc.o] Error 1
```

The reason for this error is not yet known

## Conv operator

the conv operator in the unified operator library might not be correct, so use native cpu implementation instead of the unified operator

in src/kernels/cpu/conv.cc, register the cpu conv kernel

**And**

to use the conv of CUDA(not in the unified operators library), in CMakeList.txt, add the "src/kernels/cuda/conv.cc"

```txt
# add conv conv3d in CUDA
if(USE_CUDA)
  file(GLOB_RECURSE SRC_CUDA src/cuda/*.cc src/cuda/*.cu src/kernels/cuda/conv.cc)
  list (APPEND SRC ${SRC_CUDA})
endif()
```

the reason is not yet known

## numpy version

the numpy version couldn't be too high, otherwise the result will be zero forever(examples/python/resnet_inference.py)

now the nice version is 1.26.4
