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

记录本课程的所有实践，目前进度6/12。

<!--more-->

课程主页：{{< link "https://people.maths.ox.ac.uk/gilesm/cuda/" >}}

{{< admonition warning "运行环境差异" false >}}
实验运行在不同的硬件环境下，在每个小节的开头会标注。
{{< /admonition >}}

## Practical 1: Getting Started

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 2060(6GB)
- CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
Practical 1是一个简单的“hello world”示例。

CUDA 方面：包括启动内核、将数据复制到显卡或从显卡复制数据、错误检查和从内核代码打印。
{{< /admonition >}}


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

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 4070(12GB)
- CPU：Intel(R) Core(TM) i5-13600KF CPU @ 3.40GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
受计算金融中使用蒙特卡洛方法进行期权定价的启发，我们根据独立的 "路径 "模拟，计算了 "报酬 "函数的平均值。函数的平均值。这个函数是一个随机变量，它的期望值是我们想要计算的量。具体原理见{{<link href="https://people.maths.ox.ac.uk/gilesm/cuda/prac2/MC_notes.pdf" content="some mathematical notes">}}
{{< /admonition >}}

### Subtask 1



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

但是运行速率却有较大差距：
- `__VER__=1`：17.618816ms
- `__VER__=2`：47.717342ms

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

## Practical 3: finite difference equations

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 3080(10GB)
- CPU：12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
主要目标是学习线程块优化和处理多维 PDE 应用程序的最佳方法。它还介绍了应用程序剖析的两种方法。

本实践基于雅可比迭代(Jacobi iteration)来求解三维拉普拉斯方程(3D Laplace equation)的有限差分近似值。它同时在 GPU 和 CPU 上执行计算，检查它们是否给出了相同的答案，并计算所需时间。
{{< /admonition >}}

### Subtask 1

{{< admonition type=info title="laplace3d.cu" open=false >}}
```c
//
// Program to solve Laplace equation on a regular 3D grid
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 16
#define BLOCK_Y 8

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include "laplace3d_kernel.h"

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_laplace3d(int NX, int NY, int NZ, float *h_u1, float *h_u2);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  // 'h_' prefix - CPU (host) memory space

  int NX = 1024, NY = 1024, NZ = 1024, REPEAT = 10, bx, by, i, j, k, ind;
  float *h_u1, *h_u2, *h_u3, *h_foo, err;

  // 'd_' prefix - GPU (device) memory space

  float *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  h_u2 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  h_u3 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  checkCudaErrors(cudaMalloc((void **)&d_u1, sizeof(float) * NX * NY * NZ));
  checkCudaErrors(cudaMalloc((void **)&d_u2, sizeof(float) * NX * NY * NZ));

  // initialise u1

  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        ind = i + j * NX + k * NX * NY;

        if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 ||
            k == NZ - 1)
          h_u1[ind] = 1.0f;  // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  cudaEventRecord(start);
  checkCudaErrors(cudaMemcpy(d_u1, h_u1, sizeof(float) * NX * NY * NZ,
                             cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u1 to device: %.1f (ms) \n", milli);

  // Set up the execution configuration

  bx = 1 + (NX - 1) / BLOCK_X;
  by = 1 + (NY - 1) / BLOCK_Y;

  dim3 dimGrid(bx, by);
  dim3 dimBlock(BLOCK_X, BLOCK_Y);

  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  cudaEventRecord(start);

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    getLastCudaError("GPU_laplace3d execution failed\n");

    d_foo = d_u1;
    d_u1 = d_u2;
    d_u2 = d_foo;  // swap d_u1 and d_u2
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx GPU_laplace3d: %.1f (ms) \n", REPEAT, milli);

  // Read back GPU results

  cudaEventRecord(start);
  checkCudaErrors(cudaMemcpy(h_u2, d_u1, sizeof(float) * NX * NY * NZ,
                             cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u2 to host: %.1f (ms) \n", milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // Gold treatment

  cudaEventRecord(start);
  for (int i = 1; i <= REPEAT; ++i) {
    Gold_laplace3d(NX, NY, NZ, h_u1, h_u3);
    h_foo = h_u1;
    h_u1 = h_u3;
    h_u3 = h_foo;  // swap h_u1 and h_u3
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx Gold_laplace3d: %.1f (ms) \n \n", REPEAT, milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u1[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // error check

  err = 0.0;

  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        ind = i + j * NX + k * NX * NY;
        err += (h_u1[ind] - h_u2[ind]) * (h_u1[ind] - h_u2[ind]);
      }
    }
  }

  printf("rms error = %f \n", sqrt(err / (float)(NX * NY * NZ)));

  // Release GPU and CPU memory

  checkCudaErrors(cudaFree(d_u1));
  checkCudaErrors(cudaFree(d_u2));
  free(h_u1);
  free(h_u2);
  free(h_u3);

  cudaDeviceReset();
}
```
{{< /admonition >}}

{{< admonition type=info title="laplace3d_kernel.h" open=false >}}
```h
//
// Notes:one thread per node in the 2D block;
// after initialisation it marches in the k-direction
//

// device code

__global__ void GPU_laplace3d(int NX, int NY, int NZ,
                      const float* __restrict__ d_u1,
                            float* __restrict__ d_u2)
{
  int   i, j, k, indg, active, IOFF, JOFF, KOFF;
  float u2, sixth=1.0f/6.0f;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;
  indg = i + j*NX;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  active = i>=0 && i<=NX-1 && j>=0 && j<=NY-1;

  for (k=0; k<NZ; k++) {

    if (active) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        u2 = d_u1[indg];  // Dirichlet b.c.'s
      }
      else {
        u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
             + d_u1[indg-JOFF] + d_u1[indg+JOFF]
             + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
      }
      d_u2[indg] = u2;

      indg += KOFF;
    }
  }
}
```

{{< /admonition >}}

{{< admonition type=info title="laplace3d_gold.cpp" open=false >}}
```cpp
////////////////////////////////////////////////////////////////////////////////
void Gold_laplace3d(int NX, int NY, int NZ, float* u1, float* u2) {
  int i, j, k, ind;
  float sixth =
      1.0f / 6.0f;  // predefining this improves performance more than 10%

  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX;
           i++) {  // i loop innermost for sequential memory access
        ind = i + j * NX + k * NX * NY;

        if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 ||
            k == NZ - 1) {
          u2[ind] = u1[ind];  // Dirichlet b.c.'s
        } else {
          u2[ind] = (u1[ind - 1] + u1[ind + 1] + u1[ind - NX] + u1[ind + NX] +
                     u1[ind - NX * NY] + u1[ind + NX * NY]) *
                    sixth;
        }
      }
    }
  }
}
```
{{< /admonition >}}

运行结果如下：
```
Grid dimensions: 512 x 512 x 512
Copy u1 to device: 111.1 (ms)
10x GPU_laplace3d: 41.3 (ms)
Copy u2 to host: 341.6 (ms)
10x Gold_laplace3d: 16424.9 (ms)
rms error = 0.000000
```

不难发现GPU版`GPU_laplace3d`相比于CPU版`Gold_laplace3d`快了近100倍(如果算上数据拷贝时间，则快30倍)，且精度完全一致。

GPU在xy方向分块，将每个线程块的大小设置为 $16\times 8$，共128个线程，每个线程负责“一条”节点（Z方向）的计算。因此，每个线程块的计算是独立的，不会发生冲突，因此可以并行计算。

现在将立方体大小从$512\times 512\times 512$扩展到 $1024\times 1024\times 1024$，再次运行，得到结果如下：
```
Grid dimensions: 1024 x 1024 x 1024
Copy u1 to device: 2249.3 (ms)
10x GPU_laplace3d: 476.6 (ms)
Copy u2 to host: 2949.8 (ms)
10x Gold_laplace3d: 148936.7 (ms)
rms error = 0.000000
```

数据量扩大了8倍，发现运行和拷贝时间都增加了10倍多。

调整线程块的大小，结果如下：

```
Copy u1 to device: 1808.2 (ms)
10x GPU_laplace3d: 734.5 (ms)
Copy u2 to host: 2828.9 (ms)
BLOCK_X=8 BLOCK_Y=8 : total time: 5371.6 (ms)

Copy u1 to device: 1823.6 (ms)
10x GPU_laplace3d: 476.7 (ms)
Copy u2 to host: 2886.1 (ms)
BLOCK_X=16 BLOCK_Y=8 : total time: 5186.5 (ms)

Copy u1 to device: 1811.9 (ms)
10x GPU_laplace3d: 472.7 (ms)
Copy u2 to host: 3082.9 (ms)
BLOCK_X=16 BLOCK_Y=16 : total time: 5367.6 (ms)

Copy u1 to device: 1814.0 (ms)
10x GPU_laplace3d: 485.4 (ms)
Copy u2 to host: 2821.1 (ms)
BLOCK_X=16 BLOCK_Y=32 : total time: 5120.6 (ms)

Copy u1 to device: 1813.4 (ms)
10x GPU_laplace3d: 228.1 (ms)
Copy u2 to host: 2843.5 (ms)
BLOCK_X=32 BLOCK_Y=32 : total time: 4885.0 (ms)
```

发现 $32\times 32$ 时最快（更大则无法运行），但是由于性能瓶颈出现在数据传输上，所以从总时间上看并不明显。

接下来运行一份新的GPU代码`laplace3d_new.cu`，看看与`laplace3d.cu`的差距。

{{< admonition type=info title="laplace3d_new.cu" open=false >}}
```c
//
// Program to solve Laplace equation on a regular 3D grid
//

#include <helper_cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 16
#define BLOCK_Y 4
#define BLOCK_Z 4

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel_new.h>

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_laplace3d(int NX, int NY, int NZ, float *h_u1, float *h_u2);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  // 'h_' prefix - CPU (host) memory space

  int NX = 512, NY = 512, NZ = 512, REPEAT = 10, bx, by, bz, i, j, k, ind;
  float *h_u1, *h_u2, *h_u3, *h_foo, err;

  // 'd_' prefix - GPU (device) memory space

  float *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  h_u2 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  h_u3 = (float *)malloc(sizeof(float) * NX * NY * NZ);
  checkCudaErrors(cudaMalloc((void **)&d_u1, sizeof(float) * NX * NY * NZ));
  checkCudaErrors(cudaMalloc((void **)&d_u2, sizeof(float) * NX * NY * NZ));

  // initialise u1

  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        ind = i + j * NX + k * NX * NY;

        if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 ||
            k == NZ - 1)
          h_u1[ind] = 1.0f;  // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  cudaEventRecord(start);
  checkCudaErrors(cudaMemcpy(d_u1, h_u1, sizeof(float) * NX * NY * NZ,
                             cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u1 to device: %.1f (ms) \n", milli);

  // Set up the execution configuration

  bx = 1 + (NX - 1) / BLOCK_X;
  by = 1 + (NY - 1) / BLOCK_Y;
  bz = 1 + (NZ - 1) / BLOCK_Z;

  dim3 dimGrid(bx, by, bz);
  dim3 dimBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);

  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  cudaEventRecord(start);

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    getLastCudaError("GPU_laplace3d execution failed\n");

    d_foo = d_u1;
    d_u1 = d_u2;
    d_u2 = d_foo;  // swap d_u1 and d_u2
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx GPU_laplace3d_new: %.1f (ms) \n", REPEAT, milli);

  // Read back GPU results

  cudaEventRecord(start);
  checkCudaErrors(cudaMemcpy(h_u2, d_u1, sizeof(float) * NX * NY * NZ,
                             cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\nCopy u2 to host: %.1f (ms) \n", milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // Gold treatment

  cudaEventRecord(start);
  for (int i = 1; i <= REPEAT; ++i) {
    Gold_laplace3d(NX, NY, NZ, h_u1, h_u3);
    h_foo = h_u1;
    h_u1 = h_u3;
    h_u3 = h_foo;  // swap h_u1 and h_u3
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  printf("\n%dx Gold_laplace3d: %.1f (ms) \n \n", REPEAT, milli);

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u1[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // error check

  err = 0.0;

  for (k = 0; k < NZ; k++) {
    for (j = 0; j < NY; j++) {
      for (i = 0; i < NX; i++) {
        ind = i + j * NX + k * NX * NY;
        err += (h_u1[ind] - h_u2[ind]) * (h_u1[ind] - h_u2[ind]);
      }
    }
  }

  printf("rms error = %f \n", sqrt(err / (float)(NX * NY * NZ)));

  // Release GPU and CPU memory

  checkCudaErrors(cudaFree(d_u1));
  checkCudaErrors(cudaFree(d_u2));
  free(h_u1);
  free(h_u2);
  free(h_u3);

  cudaDeviceReset();
}
```
{{< /admonition >}}

{{< admonition type=info title="laplace3d_kernel_new.h" open=false >}}
```h
//
// Notes: one thread per node in the 3D block
//

// device code

__global__ void GPU_laplace3d(int NX, int NY, int NZ,
			const float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
			      //  float* d_u1,
			      //  float* d_u2)
{
  int   i, j, k, indg, IOFF, JOFF, KOFF;
  float u2, sixth=1.0f/6.0f;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;
  k    = threadIdx.z + blockIdx.z*BLOCK_Z;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  indg = i + j*JOFF + k*KOFF;

  if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1 && k>=0 && k<=NZ-1) {
    if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
      u2 = d_u1[indg];  // Dirichlet b.c.'s
    }
    else {
      u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
           + d_u1[indg-JOFF] + d_u1[indg+JOFF]
           + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
    }
    d_u2[indg] = u2;
  }
}
```
{{< /admonition >}}

分别将立方体大小设置为 $512\times512\times512$ 和 $1024\times1024\times 1024$，运行结果如下：

```
Grid dimensions: 512 x 512 x 512
Copy u1 to device: 111.1 (ms)
10x GPU_laplace3d_new: 19.8 (ms)
Copy u2 to host: 355.3 (ms)
10x Gold_laplace3d: 16484.3 (ms)
rms error = 0.000000

Grid dimensions: 1024 x 1024 x 1024
Copy u1 to device: 1778.5 (ms)
10x GPU_laplace3d_new: 156.3 (ms)
Copy u2 to host: 2866.0 (ms)
10x Gold_laplace3d: 133816.0 (ms)
rms error = 0.000000
```

不难看出，最主要的区别在于GPU函数的运行时间，新版是旧版的1/3左右，而其他（拷贝/CPU函数）的运行时间都差不多。

两份代码的核心区别在于，`laplace3d_new`在`laplace3d`的基础上，对Z轴也进行了分块，分成了大小为 $16\times 4\times4=256$ 的线程块。

类似的，尝试调整线程块的大小，看看运行效率的变化：

```
Copy u1 to device: 2003.4 (ms)
10x GPU_laplace3d_new: 185.6 (ms)
Copy u2 to host: 3132.3 (ms)
BLOCK_X=4 BLOCK_Y=4 BLOCK_Z=4 : total time: 5321.4 (ms)

Copy u1 to device: 3127.9 (ms)
10x GPU_laplace3d_new: 152.5 (ms)
Copy u2 to host: 2852.5 (ms)
BLOCK_X=8 BLOCK_Y=8 BLOCK_Z=8 : total time: 6132.9 (ms)

Copy u1 to device: 1642.2 (ms)
10x GPU_laplace3d_new: 180.1 (ms)
Copy u2 to host: 3088.0 (ms)
BLOCK_X=16 BLOCK_Y=8 BLOCK_Z=8 : total time: 4910.2 (ms)
```

差距也不大，还是数据传输占了时间的大头。

### Subtask 2

坑代填，由于autodl的机子上没有安装`ncu`和`nsys`，等回到本机时再做。

## Practical 4: reduction operation

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 3080(10GB)
- CPU：12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
本实用程序的主要目标是了解:
- 如何使用动态大小的共享内存`shared`
- 线程同步的重要性
- 如何实现全局`reduction`，这是许多应用程序的关键要求
- 如何使用`shuffle`指令
{{< /admonition >}}

{{< admonition type=info title="reduction.cu" open=false >}}
```c
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float *idata, int len) {
  float sum = 0.0f;
  for (int i = 0; i < len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata) {
  // dynamically allocated shared memory

  extern __shared__ float temp[];

  int tid = threadIdx.x;

  // first, each thread loads data into shared memory

  temp[tid] = g_idata[tid];

  // next, we perform binary tree reduction

  for (int d = blockDim.x / 2; d > 0; d = d / 2) {
    __syncthreads();  // ensure previous step completed
    if (tid < d) temp[tid] += temp[tid + d];
  }

  // finally, first thread puts result into global memory

  if (tid == 0) g_odata[0] = temp[0];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int num_elements, num_blocks, num_threads, mem_size, shared_mem_size;

  float *h_data, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks = 1;  // start with only 1 thread block
  num_threads = 512;
  num_elements = num_blocks * num_threads;
  mem_size = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float *)malloc(mem_size);

  for (int i = 0; i < num_elements; i++)
    h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));

  // compute reference solution

  sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, sizeof(float)));

  // copy host memory to device input array

  checkCudaErrors(
      cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors(
      cudaMemcpy(h_data, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

  // check results

  printf("reduction error = %f\n", h_data[0] - sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
```
{{< /admonition >}}

`redution.cu` 是一份简单的数组求和的代码。它假设了数组长度为 $2$ 的整数次幂，且只使用1个线程块。

第一步修改是打破数组长度的假设，使它可以运行在任意长度的数组上。

只需要将`d`的初值赋为小于`blockDim.x`的最大的2的整数次幂（$2^{\lceil\log_2{\text{blockDim.x}}\rceil-1}$）。

```c
__global__ void reduction(float *g_odata, float *g_idata) {
  // dynamically allocated shared memory

  extern __shared__ float temp[];

  int tid = threadIdx.x;

  // first, each thread loads data into shared memory

  temp[tid] = g_idata[tid];

  // next, we perform binary tree reduction

  int d = 1;
  while (d * 2 <= blockDim.x) d *= 2;
  for (; d > 0; d = d / 2) {
    __syncthreads();  // ensure previous step completed
    if (tid < d && tid + d < blockDim.x) temp[tid] += temp[tid + d];
  }

  // finally, first thread puts result into global memory

  if (tid == 0) g_odata[0] = temp[0];
}
```

下一步是改变线程块的个数。注意`__shared__ float temp[]` 实在内存块内共享的。

{{< admonition type=info title="reduction_new.cu" open=false >}}
```c
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float *idata, int len) {
  float sum = 0.0f;
  for (int i = 0; i < len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata, int n) {
  // dynamically allocated shared memory

  extern __shared__ float temp[];

  int tid = threadIdx.x;

  // first, each thread loads data into shared memory

  temp[tid] = blockIdx.x * blockDim.x + tid < n
                  ? g_idata[blockIdx.x * blockDim.x + tid]
                  : 0.0f;

  // next, we perform binary tree reduction

  for (int d = blockDim.x / 2; d > 0; d = d / 2) {
    __syncthreads();  // ensure previous step completed
    if (threadIdx.x < d) temp[tid] += temp[tid + d];
  }

  // finally, first thread puts result into global memory

  if (threadIdx.x == 0) atomicAdd(&g_odata[blockIdx.x], temp[0]);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  int num_elements, num_blocks, num_threads, mem_size, shared_mem_size;

  float *h_data, *h_odata, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 260817;
  num_threads = 512;
  num_blocks = (num_elements + num_threads - 1) / num_threads;
  mem_size = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float *)malloc(mem_size);
  h_odata = (float *)malloc(num_blocks * sizeof(float));

  for (int i = 0; i < num_elements; i++)
    h_data[i] = floorf(10.0f * (rand() / (float)RAND_MAX));

  // compute reference solution

  sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, num_blocks * sizeof(float)));

  // copy host memory to device input array

  checkCudaErrors(
      cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata,
                                                          num_elements);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors(cudaMemcpy(h_odata, d_odata, num_blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));

  // check results

  float h_sum = 0.0f;
  for (int i = 0; i < num_blocks; i++) h_sum += h_odata[i];

  printf("reduction error = %f\n", h_sum - sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
```
{{< /admonition >}}

最后是使用`shuffle` 指令对块内的数据进行求和。

由于Warp Size只有32，因此如果线程块大小超过32，则还需要在块内再进行一次归约(此时块的大小可以为 $32^2=1024$，但是再大的我也跑不了了，所以就默认两层）。

```c
#define WARP_SIZE 32
__device__ __forceinline__ float warpReduceSum(float sum, int blockSize) {
  if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__global__ void reduction(float *g_odata, float *g_idata, int n) {
  int tid = threadIdx.x;

  float sum = 0;

  int i = blockIdx.x * blockDim.x + tid;
  int imax = min(n, (blockIdx.x + 1) * blockDim.x);

  while (i < imax) {
    sum += g_idata[i];
    i += blockDim.x;
  }

  extern __shared__ float warpLevelSums[];
  int laneId = threadIdx.x % WARP_SIZE;
  int warpId = threadIdx.x / WARP_SIZE;

  sum = warpReduceSum(sum, blockDim.x);

  if (laneId == 0) warpLevelSums[warpId] = sum;
  __syncthreads();
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) atomicAdd(&g_odata[blockIdx.x], sum);
}
```

## Practical 5: CUBLAS and CUFFT libraries

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 3080(10GB)
- CPU：12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
本实践介绍了 CUBLAS 和 CUFFT 库，并使用了 NVIDIA 在 GitHub 上提供的示例代码来介绍 CUBLAS 和 CUFFT 库。
{{< /admonition >}}

提供了两份示例代码，具体API可以查看文档。

{{< admonition type=info title="simpleCUBLAS.cpp" open=false >}}
```c
/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* Matrix size */
#define N  (275)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    h_C = (float *)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

    if (error_norm / ref_norm < 1e-6f)
    {
        printf("simpleCUBLAS test passed.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("simpleCUBLAS test failed.\n");
        exit(EXIT_FAILURE);
    }
}
```
{{< /admonition >}}

{{< admonition type=info title="simpleCUFFT.cu" open=false >}}
```c
/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *,
                                                   int, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleCUFFT] is starting...\n");

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  Complex *h_signal = (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // Initalize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    h_signal[i].x = rand() / (float)RAND_MAX;
    h_signal[i].y = 0;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel =
      (Complex *)malloc(sizeof(Complex) * FILTER_KERNEL_SIZE);

  // Initalize the memory for the filter
  for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
    h_filter_kernel[i].x = rand() / (float)RAND_MAX;
    h_filter_kernel[i].y = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
              &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
  int mem_size = sizeof(Complex) * new_size;

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
  // Copy host memory to device
  checkCudaErrors(
      cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

  // Allocate device memory for filter kernel
  Complex *d_filter_kernel;
  checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
                             cudaMemcpyHostToDevice));

  // CUFFT plan
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

  // Transform signal and kernel
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal,
                               (cufftComplex *)d_signal, CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_filter_kernel,
                               (cufftComplex *)d_filter_kernel, CUFFT_FORWARD));

  // Multiply the coefficients together and normalize the result
  printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
  ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size,
                                           1.0f / new_size);

  // Check if kernel execution generated and error
  getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal,
                               (cufftComplex *)d_signal, CUFFT_INVERSE));

  // Copy device memory to host
  Complex *h_convolved_signal = h_padded_signal;
  checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                             cudaMemcpyDeviceToHost));

  // Allocate host memory for the convolution result
  Complex *h_convolved_signal_ref =
      (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE);

  // Convolve on the host
  Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
           h_convolved_signal_ref);

  // check result
  bool bTestResult =
      sdkCompareL2fe((float *)h_convolved_signal_ref,
                     (float *)h_convolved_signal, 2 * SIGNAL_SIZE, 1e-5f);

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  free(h_signal);
  free(h_filter_kernel);
  free(h_padded_signal);
  free(h_padded_filter_kernel);
  free(h_convolved_signal_ref);
  checkCudaErrors(cudaFree(d_signal));
  checkCudaErrors(cudaFree(d_filter_kernel));

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();
  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
            const Complex *filter_kernel, Complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  Complex *new_data = (Complex *)malloc(sizeof(Complex) * new_size);
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = (Complex *)malloc(sizeof(Complex) * new_size);
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;

  // Loop over output element indices
  for (int i = 0; i < signal_size; ++i) {
    filtered_signal[i].x = filtered_signal[i].y = 0;

    // Loop over convolution indices
    for (int j = -maxRadius + 1; j <= minRadius; ++j) {
      int k = i + j;

      if (k >= 0 && k < signal_size) {
        filtered_signal[i] =
            ComplexAdd(filtered_signal[i],
                       ComplexMul(signal[k], filter_kernel[minRadius - j]));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
```
{{< /admonition >}}

文档见：{{<link href="https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf" content="CUBLAS" >}}、{{<link href="https://docs.nvidia.com/cuda/pdf/CUFFT_Library.pdf" content="CUFFT" >}}。

## Practical 6: odds and ends

{{< admonition type=tips title="运行环境" open=false >}}
本次实践的运行环境为：
- GPU：RTX 3080(10GB)
- CPU：12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
{{< /admonition >}}

{{< admonition type=quote title="摘要" open=true >}}
本实践的主要目标是了解
- 如何使用 g++ 编译主代码，并仅在其调用的例程中使用 CUDA
- 如何创建库文件
- 如何使用 C++ 模板
{{< /admonition >}}

{{< admonition type=info title="main.cpp" open=false >}}
```cpp
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


//
// declare external routine
//

extern
int prac6(int nblocks, int nthreads);

//
// main code
//

int main(int argc, char **argv)
{
  // set number of blocks, and threads per block

  int nblocks  = 2;
  int nthreads = 8;

  // call CUDA routine

  prac6(nblocks,nthreads);

  return 0;
}
```
{{< /admonition >}}

{{< admonition type=info title="prac6.cu" open=false >}}
```c
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = threadIdx.x;
}


//
// CUDA routine to be called by main code
//

extern
int prac6(int nblocks, int nthreads)
{
  float *h_x, *d_x;
  int   nsize, n; 

  // allocate memory for arrays

  nsize = nblocks*nthreads ;

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);

  // copy back results and print them out

  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  return 0;
}
```
{{< /admonition >}}

{{< admonition type=info title="prac6b.cu" open=false >}}
```c
//
// include files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// template kernel routine
//

template <class T>
__global__ void my_first_kernel(T *x) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  x[tid] = threadIdx.x;
}

//
// CUDA routine to be called by main code
//

extern int prac6(int nblocks, int nthreads) {
  float *h_x, *d_x;
  int *h_i, *d_i;
  double *h_y, *d_y;
  int nsize, n;

  // allocate memory for arrays

  nsize = nblocks * nthreads;

  h_x = (float *)malloc(nsize * sizeof(float));
  cudaMalloc((void **)&d_x, nsize * sizeof(float));

  h_i = (int *)malloc(nsize * sizeof(int));
  cudaMalloc((void **)&d_i, nsize * sizeof(int));

  h_y = (double *)malloc(nsize * sizeof(double));
  cudaMalloc((void **)&d_y, nsize * sizeof(double));

  // execute kernel for float

  my_first_kernel<<<nblocks, nthreads>>>(d_x);
  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  x  =  %d  %f \n", n, h_x[n]);

  // execute kernel for ints

  my_first_kernel<<<nblocks, nthreads>>>(d_i);
  cudaMemcpy(h_i, d_i, nsize * sizeof(int), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  i  =  %d  %d \n", n, h_i[n]);

  my_first_kernel<<<nblocks, nthreads>>>(d_y);
  cudaMemcpy(h_y, d_y, nsize * sizeof(double), cudaMemcpyDeviceToHost);
  for (n = 0; n < nsize; n++) printf(" n,  i  =  %d  %lf \n", n, h_y[n]);

  // free memory

  cudaFree(d_x);
  free(h_x);
  cudaFree(d_i);
  free(h_i);

  return 0;
}
```
{{< /admonition >}}

{{< admonition type=info title="prac6c.cu" open=false >}}
```c
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// template kernel routine
// 

template <int size>
__global__ void my_first_kernel(float *x)
{
  float xl[size];

  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  for (int i=0; i<size; i++) {
    xl[i] = expf((float) i*tid);
  }

  float sum = 0.0f;

  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      sum += xl[i]*xl[j];
    }
  }

  x[tid] = sum;
}


//
// CUDA routine to be called by main code
//

extern
int prac6(int nblocks, int nthreads)
{
  float *h_x, *d_x;
  int   nsize, n; 

  // allocate memory for arrays

  nsize = nblocks*nthreads ;

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel for size=2

  my_first_kernel<2><<<nblocks,nthreads>>>(d_x);
  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %g \n",n,h_x[n]);

  // execute kernel for size=3

  my_first_kernel<3><<<nblocks,nthreads>>>(d_x);
  cudaMemcpy(h_x,d_x,nsize*sizeof(int),cudaMemcpyDeviceToHost);
  for (n=0; n<nsize; n++) printf(" n,  i  =  %d  %g \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  return 0;
}
```
{{< /admonition >}}

{{< admonition type=info title="Makefile" open=false >}}
```makefile

INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math


all:	prac6 prac6a prac6b prac6c


main.o:	main.cpp Makefile
	g++ -c -fPIC -o main.o main.cpp

prac6.o:	prac6.cu Makefile
	nvcc prac6.cu -c -o prac6.o $(INC) $(NVCCFLAGS)

prac6.a:	prac6.cu Makefile
	nvcc prac6.cu -lib -o prac6.a $(INC) $(NVCCFLAGS)

prac6:	main.o prac6.o Makefile
	g++ -fPIC -o prac6 main.o prac6.o $(LIB)

prac6a:	main.cpp prac6.a Makefile
	g++ -fPIC -o prac6a main.o prac6.a $(LIB)

prac6b:	main.cpp prac6b.cu Makefile
	nvcc main.cpp prac6b.cu -o prac6b $(INC) $(NVCCFLAGS) $(LIB)

prac6c:	main.cpp prac6c.cu Makefile
	nvcc main.cpp prac6c.cu -o prac6c $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm prac6 prac6a prac6b prac6c *.o *.a
```
{{< /admonition >}}

如果`make prac6` 时报错：`/usr/bin/ld: cannot find -lcudart`，说明`libcudart.so`没有链接上。到`cuda`的文件夹内找到`../targets/x86_64-linux/lib/libcudart.so`，然后软连接到`/usr/local/lib/libcudart.so`中。

`prac6` 和 `prac6a` 展示了两种将`cu`文件链接到`cpp`中的方式。
- `g++ -fPIC -o prac6 main.o prac6.o $(LIB)`：生成的对象文件会链接到 `main.o` 以创建 `prac6`
- `g++ -fPIC -o prac6a main.o prac6.a $(LIB)`：被编译成一个库，然后再链接到 `main.o`

`prac6b` 展示了如何使用`template<class T>`，从而给函数传入不同类型的数据。

`prac6c` 展示了如何使用`template <int size>`，从而在函数内生成确定大小的数组，编译器会将固定大小的数组映射到寄存器中。

更多关于`nvcc`编译器的内容见文档：{{<link "http://docs.nvidia.com/cuda/pdf/CUDA Compiler Driver NVCC.pdf">}}

