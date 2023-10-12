---
title: "Oxford: Course on CUDA Programming on NVIDIA GPUs Practice"
date: 2023-10-11T20:15:04+08:00
draft: false
---

记录本课程的所有practice。

目前进度1/12。

<!--more-->

课程主页：{{< link "https://people.maths.ox.ac.uk/gilesm/cuda/" >}}

## Practical 1: Getting Started

Practice 1是一个简单的“hello world”示例。

CUDA 方面：包括启动内核、将数据复制到显卡或从显卡复制数据、错误检查和从内核代码打印。

给了三份cuda代码，包括`prac1a.cu`、`prac1b.cu`和`prac1c.cu`。

{{< admonition type=info title="prac1a.cu" open=false >}}

```cuda
//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// kernel routine
//

__global__ void my_first_kernel(float *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float)threadIdx.x;
}

//
// main code
//

int main(int argc, char **argv) {
  float *h_x, *d_x;
  int nblocks, nthreads, nsize, n;

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 4;
  nsize = nblocks * nthreads;

  // allocate memory for array

  h_x = (float *)malloc(nsize * sizeof(float));
  cudaMalloc((void **)&d_x, nsize * sizeof(float));

  // execute kernel

  my_first_kernel<<<nblocks, nthreads>>>(d_x);

  // copy back results and print them out

  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);

  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

  // free memory

  cudaFree(d_x);
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}

```

{{< /admonition >}}

{{< admonition type=info title="prac1b.cu" open=false >}}

```cuda
//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

//
// kernel routine
//

__global__ void my_first_kernel(float *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float)threadIdx.x;
}

//
// main code
//

int main(int argc, const char **argv) {
  float *h_x, *d_x;
  int nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 8;
  nsize = nblocks * nthreads;

  // allocate memory for array

  h_x = (float *)malloc(nsize * sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize * sizeof(float)));

  // execute kernel

  my_first_kernel<<<nblocks, nthreads>>>(d_x);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors(
      cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost));

  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

  // free memory

  checkCudaErrors(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}


```

{{< /admonition >}}

`prac1a.cu`和`prac1b.cu`的功能相同，都使用2个block，每个block分配4个thread。每个thread计算出它在全局中的下标(`threadIdx.x + blockDim.x * blockIdx.x`)，然后将threadId.x的赋到数组x中。

区别在于`prac1b.cu`在`prac1a.cu`的基础上增加了`checkCudaErrors`和`getLastCudaError`函数，用于检验运行时的错误。

运行结果如下：

```plain
 n,  x  =  0  0.000000
 n,  x  =  1  1.000000
 n,  x  =  2  2.000000
 n,  x  =  3  3.000000
 n,  x  =  4  0.000000
 n,  x  =  5  1.000000
 n,  x  =  6  2.000000
 n,  x  =  7  3.000000
```

适当增大`nblocks`和`nthreads`，例如 `nblocks=200` 和 `nthreads=400` ，运行结果正常，但是如果继续增大，例如`nblocks=2000` 和 `nthreads=4000`，运行时则会出现以下错误：
```
prac1b.cu(48) : getLastCudaError() CUDA error : my_first_kernel execution failed
 : (9) invalid configuration argument.
```

原因在于申请的线程数量超过了硬件限制，通过`cudaDeviceGetAttribute`函数，即可查询本机的硬件限制信息。

{{< admonition type=info title="getInfo.cu" open=false >}}
```cuda
#include <cuda_runtime.h>

#include <iostream>

int main() {
  int deviceId = 0;  // 选择要查询的CUDA设备的ID，这里假设使用设备0

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  int maxBlocksPerSM;
  int maxThreadsPerBlock;
  int maxThreadsPerSM;

  // 查询最大可分配线程块数量
  cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor,
                         deviceId);

  // 查询线程块中的线程数量
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock,
                         deviceId);

  // 计算nsize（nblocks * nthreads）
  maxThreadsPerSM = maxBlocksPerSM * maxThreadsPerBlock;

  std::cout << "Max Blocks Per SM: " << maxBlocksPerSM << std::endl;
  std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads Per SM (nsize): " << maxThreadsPerSM << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor
            << std::endl;

  return 0;
}


```
{{< /admonition >}}

编译运行`getInfo.cu`，输出为：
{{< highlight plain "hl_lines=2" >}}
Max Blocks Per SM: 16
Max Threads Per Block: 1024
Max Threads Per SM (nsize): 16384
Device name: NVIDIA GeForce RTX 2060
Compute capability: 7.5
{{< /highlight >}}

不过经过测试发现，只要保证`nthreads`不超过$1024$，随意调整`nblocks`的大小都不会出现错误（即使达到百万）。

当然，进行同样的修改后运行`prac1a`，则不会有任何报错，但是输出结果全是0。显然代码在运行中出现了问题，但我们只有在观察到输出不符合预期才能发觉，这也说明了`checkCudaErrors`等检查函数的意义。

下一步是修改`prac1b.cu`，使它实现两个数组的逐项求和。

基本思路就是先申请三个数组，分别存储两个数组的值和结果，然后在kernel中进行计算，求和时传入三个数组的指针，最后将结果拷贝回主机并输出。

{{< admonition type=info title="prac1b_new.cu" open=false >}}

```cuda
//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

//
// kernel routine
//

__global__ void init(float *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float)threadIdx.x;
}

__global__ void add(float *x, float *y, float *z) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  z[tid] = x[tid] + y[tid];
}

//
// main code
//

int main(int argc, const char **argv) {
  float *h_x, *d_x1, *d_x2, *d_x3;
  int nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 4;
  nsize = nblocks * nthreads;

  // allocate memory for array

  h_x = (float *)malloc(nsize * sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x1, nsize * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x2, nsize * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_x3, nsize * sizeof(float)));

  // execute kernel

  init<<<nblocks, nthreads>>>(d_x1);
  getLastCudaError("init execution failed\n");
  init<<<nblocks, nthreads>>>(d_x2);
  getLastCudaError("init execution failed\n");
  add<<<nblocks, nthreads>>>(d_x1, d_x2, d_x3);
  getLastCudaError("add execution failed\n");

  // copy back results and print them out

  checkCudaErrors(
      cudaMemcpy(h_x, d_x3, nsize * sizeof(float), cudaMemcpyDeviceToHost));

  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

  // free memory

  checkCudaErrors(cudaFree(d_x1));
  checkCudaErrors(cudaFree(d_x2));
  checkCudaErrors(cudaFree(d_x3));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
```
{{< /admonition >}}

结果如下：

```plain
 n,  x  =  0  0.000000
 n,  x  =  1  2.000000
 n,  x  =  2  4.000000
 n,  x  =  3  6.000000
 n,  x  =  4  0.000000
 n,  x  =  5  2.000000
 n,  x  =  6  4.000000
 n,  x  =  7  6.000000
```

每个元素的值是上次的两倍，说明正确的进行了求和。

这份代码考虑的情况比较简单，更加完整的实现可以参考官方的示例代码[vectorAdd.cu](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu)。

{{< admonition type=info title="prac1c.cu" open=false >}}
```cuda
//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

//
// kernel routine
//

__global__ void my_first_kernel(float *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = (float)threadIdx.x;
}

//
// main code
//

int main(int argc, const char **argv) {
  float *x;
  int nblocks, nthreads, nsize, n;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks = 2;
  nthreads = 8;
  nsize = nblocks * nthreads;

  // allocate memory for array

  checkCudaErrors(cudaMallocManaged(&x, nsize * sizeof(float)));

  // execute kernel

  my_first_kernel<<<nblocks, nthreads>>>(x);
  getLastCudaError("my_first_kernel execution failed\n");

  // synchronize to wait for kernel to finish, and data copied back

  cudaDeviceSynchronize();

  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, x[n]);

  // free memory

  checkCudaErrors(cudaFree(x));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}

```
{{< /admonition >}}

相比于`prac1a.cu`和`prac1b.cu`，`prac1c.cu`的最大区别在于使用`cudaMallocManaged`函数进行内存分配。

相比于`cudaMalloc`，`cudaMallocManaged`更加智能，不再需要显式的拷贝，简化了代码的编写。

特别需要注意的是，虽然不再需要`cudaMemcpy`，但是在执行`printf`前，需要使用`cudaDeviceSynchronize`函数，用于等待GPU全部线程执行完毕。