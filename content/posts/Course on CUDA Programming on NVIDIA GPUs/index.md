---
title: "Oxford: Course on CUDA Programming on NVIDIA GPUs 学习笔记"
date: 2023-10-11T20:15:04+08:00
draft: false
tags: ["CUDA"]
categories: ["学习笔记"]
resources:
- name: "featured-image"
  src: "cuda.png"
- name: "featured-image-preview"
  src: "cuda.png"
lightgallery: true
---

记录本课程的所有practice。

目前进度1/12。

<!--more-->

课程主页：{{< link "https://people.maths.ox.ac.uk/gilesm/cuda/" >}}

{{< admonition bug "运行环境差异" false >}}
实验分散在两台电脑中完成（宿舍与工位），显卡分别为2060和4070，因此实验结果（尤其是运行效率）会有差异。
{{< /admonition >}}

## Practical 1: Getting Started

Practical 1是一个简单的“hello world”示例。

CUDA 方面：包括启动内核、将数据复制到显卡或从显卡复制数据、错误检查和从内核代码打印。

给了三份cuda代码，包括`prac1a.cu`、`prac1b.cu`和`prac1c.cu`。

{{< admonition type=info title="prac1a.cu" open=false >}}

```c
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

```c
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
```c
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

不过经过测试（本机RTX2060）发现，只要保证`nthreads`不超过$1024$，随意调整`nblocks`的大小都不会出现错误（即使达到百万）。

当然，进行同样的修改后运行`prac1a`，则不会有任何报错，但是输出结果全是0。显然代码在运行中出现了问题，但我们只有在观察到输出不符合预期才能发觉，这也说明了`checkCudaErrors`等检查函数的意义。

下一步是修改`prac1b.cu`，使它实现两个数组的逐项求和。

基本思路就是先申请三个数组，分别存储两个数组的值和结果，然后在kernel中进行计算，求和时传入三个数组的指针，最后将结果拷贝回主机并输出。

{{< admonition type=info title="prac1b_new.cu" open=false >}}

```c
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
```c
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

## Practical 2: Monte Carlo Simulation

### Subtask 1

受计算金融中使用蒙特卡洛方法进行期权定价的启发，我们根据独立的 "路径 "模拟，计算了 "报酬 "函数的平均值。函数的平均值。这个函数是一个随机变量，它的期望值是我们想要计算的量。具体原理见{{<link href="https://people.maths.ox.ac.uk/gilesm/cuda/prac2/MC_notes.pdf" content="some mathematical notes">}}

首先是运行`prac2.cu`（相比于原版代码，我将其修改为根据宏`__VER__`来决定执行语句），感受`__constant__`的使用，以及两份代码不同的运行效率。

{{< admonition type=info title="prac2.cu" open=false >}}
```c
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "helper_cuda.h"

#define __VER__ 1

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void pathcalc(float *d_z, float *d_v) {
  float s1, s2, y1, y2, payoff;
  int ind;

  // move array pointers to correct position

#if __VER__ == 1
  ind = threadIdx.x + 2 * N * blockIdx.x * blockDim.x;
#elif __VER__ == 2
  ind = 2 * N * threadIdx.x + 2 * N * blockIdx.x * blockDim.x;
#endif
  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n = 0; n < N; n++) {
    y1 = d_z[ind];
#if __VER__ == 1
    ind += blockDim.x;  // shift pointer to next element
#elif __VER__ == 2
    ind += 1;
#endif
    y2 = rho * y1 + alpha * d_z[ind];
#if __VER__ == 1
    ind += blockDim.x;  // shift pointer to next element
#elif __VER__ == 2
    ind += 1;
#endif
    s1 = s1 * (con1 + con2 * y1);
    s2 = s2 * (con1 + con2 * y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if (fabs(s1 - 1.0f) < 0.1f && fabs(s2 - 1.0f) < 0.1f) payoff = exp(-r * T);

  d_v[threadIdx.x + blockIdx.x * blockDim.x] = payoff;
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int NPATH = 9600000, h_N = 100;
  float h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float *h_v, *d_v, *d_z;
  double sum1, sum2;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float) * NPATH);

  checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float) * NPATH));
  checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float) * 2 * h_N * NPATH));

  // define constants and transfer to GPU

  h_T = 1.0f;
  h_r = 0.05f;
  h_sigma = 0.1f;
  h_rho = 0.5f;
  h_alpha = sqrt(1.0f - h_rho * h_rho);
  h_dt = 1.0f / h_N;
  h_con1 = 1.0f + h_r * h_dt;
  h_con2 = sqrt(h_dt) * h_sigma;

  checkCudaErrors(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));
  checkCudaErrors(cudaMemcpyToSymbol(T, &h_T, sizeof(h_T)));
  checkCudaErrors(cudaMemcpyToSymbol(r, &h_r, sizeof(h_r)));
  checkCudaErrors(cudaMemcpyToSymbol(sigma, &h_sigma, sizeof(h_sigma)));
  checkCudaErrors(cudaMemcpyToSymbol(rho, &h_rho, sizeof(h_rho)));
  checkCudaErrors(cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(h_alpha)));
  checkCudaErrors(cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt)));
  checkCudaErrors(cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)));
  checkCudaErrors(cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)));

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  cudaEventRecord(start);
  checkCudaErrors(curandGenerateNormal(gen, d_z, 2 * h_N * NPATH, 0.0f, 1.0f));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
         milli, 2.0 * h_N * NPATH / (0.001 * milli));

  // execute kernel and time it

  cudaEventRecord(start);

  pathcalc<<<NPATH / 128, 128>>>(d_z, d_v);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n", milli);

  // copy back results

  checkCudaErrors(
      cudaMemcpy(h_v, d_v, sizeof(float) * NPATH, cudaMemcpyDeviceToHost));

  float *h_z = (float *)malloc(sizeof(float) * 2 * h_N * NPATH);
  checkCudaErrors(cudaMemcpy(h_z, d_z, sizeof(float) * 2 * h_N * NPATH,
                             cudaMemcpyDeviceToHost));

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i = 0; i < NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i] * h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
         sum1 / NPATH,
         sqrt((sum2 / NPATH - (sum1 / NPATH) * (sum1 / NPATH)) / NPATH));

  // Tidy up library

  checkCudaErrors(curandDestroyGenerator(gen));

  // Release memory and exit cleanly
  free(h_z);
  free(h_v);
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_z));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
```
{{< /admonition >}}

运行结果皆为：`Average value and standard deviation of error  =    0.41793859    0.00015237`，说明两份代码的正确性没有问题。

但是运行速率却有较大差距，在本机（RTX 4070）下：
- 版本1：`Monte Carlo kernel execution time (ms): 17.618816`
- 版本2：`Monte Carlo kernel execution time (ms): 47.717342`

两份版本的代码的差异在于 `pathcalc` 函数中，访问元素的顺序。

举个例子：

```c
int N = 16;
int blockDim.x = 4; // 假设每个线程块中包含4个线程
int blockIdx.x = 2; // 假设当前线程块的索引为2

// 版本1的计算方式
ind = threadIdx.x + 2 * N * blockIdx.x * blockDim.x;

// 计算每个线程的 ind
// 线程0: 0 + 2 * 16 * 2 * 4 = 128
// 线程1: 1 + 2 * 16 * 2 * 4 = 129
// 线程2: 2 + 2 * 16 * 2 * 4 = 130
// 线程3: 3 + 2 * 16 * 2 * 4 = 131

// 版本2的计算方式
ind = 2 * N * threadIdx.x + 2 * N * blockIdx.x * blockDim.x;

// 计算每个线程的 ind
// 线程0: 2 * 16 * 0 + 2 * 16 * 2 * 4 = 0
// 线程1: 2 * 16 * 1 + 2 * 16 * 2 * 4 = 144
// 线程2: 2 * 16 * 2 + 2 * 16 * 2 * 4 = 288
// 线程3: 2 * 16 * 3 + 2 * 16 * 2 * 4 = 432
```

不难看出，在版本1中，同一个block中的不同线程，访问的下标`ind`是连续的（128,129,130,131），而在版本2中，访问的下标`ind`是交替的（0,144,288,432），因此前者的效率更高。

### Subtask 2

接下来是自己实现一份代码，计算 $az^2+bz+c$的平均值（提示：结果约为$a+c$）。

参考`prac2.cu`以及`cuRAND`库的[文档](https://docs.nvidia.com/cuda/curand/index.html)、`thrust`库的示例，实现的代码如下。

{{< admonition type=info title="prac2_new.cu" open=false >}}
```c
#include <curand.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "helper_cuda.h"

__constant__ int n;
__constant__ float a, b, c;
__device__ double sum;

__global__ void calc1(float *d_z, float *d_v) {
  int ind = threadIdx.x + n * blockIdx.x * blockDim.x;
  for (int i = 0; i < n; i++) {
    float tmp = a * d_z[ind] * d_z[ind] + b * d_z[ind] + c;
    atomicAdd(&d_v[i + blockIdx.x], tmp);
    ind += blockDim.x;
  }
}

__global__ void calc2(float *d_z) {
  int ind = threadIdx.x + n * blockIdx.x * blockDim.x;
  for (int i = 0; i < n; i++) {
    float tmp = a * d_z[ind] * d_z[ind] + b * d_z[ind] + c;
    atomicAdd(&sum, tmp);
    ind += blockDim.x;
  }
}

struct MyFunction {
  float a, b, c;
  MyFunction(float _a, float _b, float _c) : a(_a), b(_b), c(_c) {}

  __host__ __device__ float operator()(const float &x) const {
    return a * x * x + b * x + c;
  }
};

void getTime(cudaEvent_t &start, cudaEvent_t &stop, const char *msg = NULL) {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milli;
  cudaEventElapsedTime(&milli, start, stop);
  if (msg != NULL) {
    cudaDeviceSynchronize();
    printf("%s: %f ms", msg, milli);
  }
}

int main(int argc, const char **argv) {
  // initialise card
  findCudaDevice(argc, argv);

  int h_N = 1280000, h_n = 1000;
  float *d_z, *d_v;
  float h_a = 1.0f, h_b = 2.0f, h_c = 3.0f;
  checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float) * h_N * h_n));
  checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float) * h_N));

  // random number generation

  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  checkCudaErrors(curandGenerateNormal(gen, d_z, h_N * h_n, 0.0f, 1.0f));
  checkCudaErrors(cudaMemcpyToSymbol(n, &h_n, sizeof(h_n)));
  checkCudaErrors(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
  checkCudaErrors(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
  checkCudaErrors(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));
  int type = 4;
  printf("a = %f, b = %f, c = %f\n", h_a, h_b, h_c);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (type == 1) {
    cudaEventRecord(start);
    calc1<<<h_N / 128, 128>>>(d_z, d_v);
    float *h_v = (float *)malloc(sizeof(double) * h_N);
    double h_sum = 0;
    checkCudaErrors(
        cudaMemcpy(h_v, d_v, sizeof(float) * h_N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < h_N; i++) {
      h_sum += h_v[i];
    }

    printf("sum = %lf\n", h_sum / (h_N * h_n));
    free(h_v);
    getTime(start, stop, "Method 1");
  }
  if (type == 2) {
    cudaEventRecord(start);
    calc2<<<h_N / 128, 128>>>(d_z);
    double h_sum;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_sum, sum, sizeof(sum)));
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 2");
  }
  if (type == 3) {
    cudaEventRecord(start);
    MyFunction f(h_a, h_b, h_c);
    double h_sum = thrust::transform_reduce(
        thrust::device, d_z, d_z + h_N * h_n, f, 0.0, thrust::plus<float>());
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 3");
  }
  if (type == 4) {
    cudaEventRecord(start);
    double h_sum = 0;
    float *h_z = (float *)malloc(sizeof(float) * h_N * h_n);
    checkCudaErrors(cudaMemcpy(h_z, d_z, sizeof(float) * h_N * h_n,
                               cudaMemcpyDeviceToHost));
    for (int i = 0; i < h_N * h_n; i++) {
      h_sum += h_a * h_z[i] * h_z[i] + h_b * h_z[i] + h_c;
    }
    printf("sum = %lf\n", h_sum / (h_N * h_n));
    getTime(start, stop, "Method 4");
  }
  checkCudaErrors(curandDestroyGenerator(gen));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_v));
  return 0;
}
```
{{< /admonition >}}

将 $a,b,c$ 分别设置为 $1.0,2.0,3.0$，运行结果期望为 $4.0$，实际结果都为 $3.99999$ 左右，验证了代码的正确性。

先学习一下如何计算运行时间：

{{< admonition type=info title="getTime.cu" open=false >}}
```c
void getTime(void (*func)(), const char *msg = NULL) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  func();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milli;
  cudaEventElapsedTime(&milli, start, stop);
  if (msg != NULL) {
    printf("%s: %f ms", msg, milli);
  }
}
```
{{< /admonition >}}

方法1：每个线程计算一个元素，对每个线程块维护元素和，最后再用CPU求出总和。

方法2：每个线程计算一个元素，直接累加到全局的`sum`变量。

方法3：使用`thrust`库，直接计算总和。

方法4：直接CPU计算总和。

三个方法的效率有较大差异：
- `type=1`：231.914246 ms
- `type=2`：1735.252563 ms
- `type=3`：11.132544 ms
- `type=4`：2783.145996 ms

由于方法3是调用的库函数，暂且不算。方法2较慢的原因可能在于由于使用`atomicAdd`，因此每个线程在更新`sum`前都要等待前一个线程完成计算，这样几乎等价于单线程计算（和方法4差距不到50%）。而在方法1中，每个线程块都有一个`sum`变量，因此相对来说冲突较少，效率更高。

### Subtask 3

最后是运行`prac2_device.cu`，学习如何使用`cuRAND`库中的`curand_init`和`curandState_t`，对于每个线程生成随机数据（而不是像之前一样，先一次性生成整个数组）。

{{< admonition type=info title="prac2_device.cu" open=false >}}
```c
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void pathcalc(float *d_v) {
  float s1, s2, y1, y2, payoff;

  curandState_t state;

  // RNG initialisation with skipahead

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(1234, id, 0, &state);

  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n = 0; n < N; n++) {
    y1 = curand_normal(&state);
    y2 = rho * y1 + alpha * curand_normal(&state);

    s1 = s1 * (con1 + con2 * y1);
    s2 = s2 * (con1 + con2 * y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if (fabs(s1 - 1.0f) < 0.1f && fabs(s2 - 1.0f) < 0.1f) payoff = exp(-r * T);

  d_v[threadIdx.x + blockIdx.x * blockDim.x] = payoff;
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int NPATH = 9600000, h_N = 100;
  float h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float *h_v, *d_v;
  double sum1, sum2;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float) * NPATH);
  checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float) * NPATH));

  // define constants and transfer to GPU

  h_T = 1.0f;
  h_r = 0.05f;
  h_sigma = 0.1f;
  h_rho = 0.5f;
  h_alpha = sqrt(1.0f - h_rho * h_rho);
  h_dt = 1.0f / h_N;
  h_con1 = 1.0f + h_r * h_dt;
  h_con2 = sqrt(h_dt) * h_sigma;

  checkCudaErrors(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));
  checkCudaErrors(cudaMemcpyToSymbol(T, &h_T, sizeof(h_T)));
  checkCudaErrors(cudaMemcpyToSymbol(r, &h_r, sizeof(h_r)));
  checkCudaErrors(cudaMemcpyToSymbol(sigma, &h_sigma, sizeof(h_sigma)));
  checkCudaErrors(cudaMemcpyToSymbol(rho, &h_rho, sizeof(h_rho)));
  checkCudaErrors(cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(h_alpha)));
  checkCudaErrors(cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt)));
  checkCudaErrors(cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)));
  checkCudaErrors(cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)));

  // execute kernel and time it

  cudaEventRecord(start);

  pathcalc<<<NPATH / 128, 128>>>(d_v);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n", milli);

  // copy back results

  checkCudaErrors(
      cudaMemcpy(h_v, d_v, sizeof(float) * NPATH, cudaMemcpyDeviceToHost));

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i = 0; i < NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i] * h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
         sum1 / NPATH,
         sqrt((sum2 / NPATH - (sum1 / NPATH) * (sum1 / NPATH)) / NPATH));

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors(cudaFree(d_v));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
```
{{< /admonition >}}

又或者根据文档，将生成随机数据和计算分成两个函数，如下：

{{< admonition type=info title="prac2_device_new.cu" open=false >}}
```c
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

__global__ void setup_kernel(curandState *state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(1234, id, 0, &state[id]);
}

__global__ void pathcalc(float *d_v, curandState *state) {
  float s1, s2, y1, y2, payoff;

  // RNG initialisation with skipahead

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];

  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n = 0; n < N; n++) {
    y1 = curand_normal(&localState);
    y2 = rho * y1 + alpha * curand_normal(&localState);

    s1 = s1 * (con1 + con2 * y1);
    s2 = s2 * (con1 + con2 * y2);
  }

  // put payoff value into device array

  payoff = 0.0f;
  if (fabs(s1 - 1.0f) < 0.1f && fabs(s2 - 1.0f) < 0.1f) payoff = exp(-r * T);

  d_v[threadIdx.x + blockIdx.x * blockDim.x] = payoff;
}
////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int NPATH = 9600000, h_N = 100;
  float h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float *h_v, *d_v;
  double sum1, sum2;
  curandState *state;
  int totalThreads = NPATH;
  cudaMalloc((void **)&state, totalThreads * sizeof(curandState));

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float) * NPATH);
  checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float) * NPATH));

  // define constants and transfer to GPU

  h_T = 1.0f;
  h_r = 0.05f;
  h_sigma = 0.1f;
  h_rho = 0.5f;
  h_alpha = sqrt(1.0f - h_rho * h_rho);
  h_dt = 1.0f / h_N;
  h_con1 = 1.0f + h_r * h_dt;
  h_con2 = sqrt(h_dt) * h_sigma;

  checkCudaErrors(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));
  checkCudaErrors(cudaMemcpyToSymbol(T, &h_T, sizeof(h_T)));
  checkCudaErrors(cudaMemcpyToSymbol(r, &h_r, sizeof(h_r)));
  checkCudaErrors(cudaMemcpyToSymbol(sigma, &h_sigma, sizeof(h_sigma)));
  checkCudaErrors(cudaMemcpyToSymbol(rho, &h_rho, sizeof(h_rho)));
  checkCudaErrors(cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(h_alpha)));
  checkCudaErrors(cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt)));
  checkCudaErrors(cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)));
  checkCudaErrors(cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)));

  // execute kernel and time it
  cudaEventRecord(start);
  int blockSize = 256;
  int gridSize = NPATH / blockSize;
  setup_kernel<<<gridSize, blockSize>>>(state);
  pathcalc<<<gridSize, blockSize>>>(d_v, state);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n", milli);

  // copy back results

  checkCudaErrors(
      cudaMemcpy(h_v, d_v, sizeof(float) * NPATH, cudaMemcpyDeviceToHost));

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i = 0; i < NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i] * h_v[i];
  }

  printf(
      "\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n ",
      sum1 / NPATH,
      sqrt((sum2 / NPATH - (sum1 / NPATH) * (sum1 / NPATH)) / NPATH));

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors(cudaFree(d_v));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
```
{{< /admonition >}}

两份代码的运行时间分别为78.679070ms和82.948097ms，相比于此前方法1的231.914246 ms，效率提升了近2倍。

文档最后还提示可以使用`cudaGetDevice`、`cudaGetDeviceProperties`和`cudaOccupancyMaxActiveBlocksPerMultiprocessor`来计算最佳的线程块大小。我将线程块大小依次调整为64、128、256、512和1024，运行时间的变化不大，都在80-100ms之间，并没有得到很大的提升。

因此我没有理解最后的`Compile and run the code to see the improved performance it gives.`的提升在哪里。。