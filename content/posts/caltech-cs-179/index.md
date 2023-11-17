---
title: "Caltech CS179 GPU Programming 学习笔记"
date: 2023-10-27T17:47:20+08:00
draft: false
tags: ["CUDA"]
categories: ["学习笔记"]
resources:
- name: "featured-image"
  src: "main.png"
- name: "featured-image-preview"
  src: "main.png"
lightgallery: true
---

记录本课程的所有Lab，目前进度1/5。

<!--more-->

## Lab 1: Introduction to CUDA

### Question 1: Common Errors (20 points)

#### 1.1

问题: 指针需要先申请内存再使用。

修正:
```c
void test1_fixed() {
    int *a = (int *) malloc(sizeof (int)); // allocate memory for a pointer
    *a = 3;
    *a = *a + 2;
    printf("%d\n", *a);
}
```

#### 1.2

问题: 指针声明的时候需要加`*`，即使在同一行申请多个指针。

修正:
```c
void test2_fixed() {
    int *a, *b; // *name to declare a pointer
    a = (int *) malloc(sizeof (int));
    b = (int *) malloc(sizeof (int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}
```

#### 1.3

问题: `malloc`的时候需要申请的内存大小为长度$\times$ 类型大小。

修正:
```c
void test3_fixed() {
    int i, *a = (int *) malloc(1000 * sizeof(int)); // multiply by sizeof(type)

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}
```

#### 1.4

问题: 二维数组的每一行都需要单独申请内存。

修正:
```c
void test4_fixed() {
    int **a = (int **) malloc(3 * sizeof (int *));
    for (int i = 0; i < 3; i++)
        a[i] = (int *) malloc(100 * sizeof(int)); // allocate memory for each row
    a[1][1] = 5;
}
```

#### 1.5

问题: `scanf`需要传入指针的地址。

修正:
```c
void test5_fixed() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", &a); // pass in the address of a
    if (!a)
        printf("Value is 0\n");
}
```

### Question 2: Parallelization (30 points)

#### 2.1

- `y_1[n] = x[n - 1] + x[n] + x[n + 1]`
- `y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n]`

第一个更容易并行化，因为`y_1`只依赖于`x`，所以每个线程都可以独立计算。

#### 2.2

- `y[n] = c * x[n] + (1 - c) * y[n - 1]`

由于 $c$ 很接近 $1$，所以 $(1 - c)$ 很接近 $0$。我们可以将 $y[n - 1]$ 展开为 $y[n - 2]$，甚至 $y[n - 3]$，后续的项则可以几乎忽略。

- `y[n] = c * x[n] + (1 - c) * (c * x[n - 1] + (1 - c)y[n - 2])` $\Rightarrow$
    - `y[n] = c * x[n] + (1 - c) * (c * x[n - 1] + (1 - c)(c * x[n - 2] + (1 - c)y[n - 3]))`

$(1 - c) ^ 2$ 很接近 $0$，所以后续的项则可以几乎忽略。

最终式子为 `y[n] = c * x[n] + (1 - c) * c * x[n - 1]`，这样就可以并行化了。

### Question 3: Small-Kernel Convolution (50 points)

{{< admonition type=info title="blur.cu" open=false >}}
```c
/*
 * CUDA blur
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#include <cuda_runtime.h>

#include <cstdio>

#include "blur.cuh"
#include "cuda_header.cuh"

CUDA_CALLABLE
void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
  // TODO: Implement the necessary convolution function that should be
  //       completed for each thread_index. Use the CPU implementation in
  //       blur.cpp as a reference.
  int jmax = thread_index >= blur_v_size ? blur_v_size : thread_index + 1;
  for (int j = 0; j < jmax; ++j) {
    gpu_out_data[thread_index] +=
        gpu_raw_data[thread_index - j] * gpu_blur_v[j];
  }
}

__global__ void cuda_blur_kernel(const float* gpu_raw_data,
                                 const float* gpu_blur_v, float* gpu_out_data,
                                 int n_frames, int blur_v_size) {
  // TODO: Compute the current thread index.
  uint thread_index = threadIdx.x + blockIdx.x * blockDim.x;

  // TODO: Update the while loop to handle all indices for this thread.
  //       Remember to advance the index as necessary.
  while (thread_index < n_frames) {
    // Do computation for this thread index
    cuda_blur_kernel_convolution(thread_index, gpu_raw_data, gpu_blur_v,
                                 gpu_out_data, n_frames, blur_v_size);
    // TODO: Update the thread index
    thread_index += blockDim.x * gridDim.x;
  }
}

float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float* raw_data, const float* blur_v,
                            float* out_data, const unsigned int n_frames,
                            const unsigned int blur_v_size) {
  // Use the CUDA machinery for recording time
  cudaEvent_t start_gpu, stop_gpu;
  float time_milli = -1;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventRecord(start_gpu);

  // TODO: Allocate GPU memory for the raw input data (either audio file
  //       data or randomly generated data. The data is of type float and
  //       has n_frames elements. Then copy the data in raw_data into the
  //       GPU memory you allocated.
  float* gpu_raw_data;
  cudaMalloc((void**)&gpu_raw_data, sizeof(float) * n_frames);
  cudaMemcpy(gpu_raw_data, raw_data, sizeof(float) * n_frames,
             cudaMemcpyHostToDevice);

  // TODO: Allocate GPU memory for the impulse signal (for now global GPU
  //       memory is fine. The data is of type float and has blur_v_size
  //       elements. Then copy the data in blur_v into the GPU memory you
  //       allocated.
  float* gpu_blur_v;
  cudaMalloc((void**)&gpu_blur_v, sizeof(float) * blur_v_size);
  cudaMemcpy(gpu_blur_v, blur_v, sizeof(float) * blur_v_size,
             cudaMemcpyHostToDevice);

  // TODO: Allocate GPU memory to store the output audio signal after the
  //       convolution. The data is of type float and has n_frames elements.
  //       Initialize the data as necessary.
  float* gpu_out_data;
  cudaMalloc((void**)&gpu_out_data, sizeof(float) * n_frames);
  cudaMemset(gpu_out_data, 0, sizeof(float) * n_frames);

  // TODO: Appropriately call the kernel function.

  cuda_blur_kernel<<<blocks, threads_per_block>>>(
      gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);

  // Check for errors on kernel call
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
  else
    fprintf(stderr, "No kernel error detected\n");

  // TODO: Now that kernel calls have finished, copy the output signal
  //       back from the GPU to host memory. (We store this channel's result
  //       in out_data on the host.)

  cudaMemcpy(out_data, gpu_out_data, sizeof(float) * n_frames,
             cudaMemcpyDeviceToHost);

  // TODO: Now that we have finished our computations on the GPU, free the
  //       GPU resources.

  cudaFree(gpu_raw_data);
  cudaFree(gpu_blur_v);
  cudaFree(gpu_out_data);

  // Stop the recording timer and return the computation time
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
  return time_milli;
}
```
{{< /admonition >}}

`TODO`写的还是挺明确的，需要补全的代码基本就是一些最基本的语法，例如`cudaMemcpy`、`cudaFree`。

核函数是实现一个线程的卷积操作，这部分可以参考CPU版的代码。

最终结果：
- **autio-blur**：设置1024个线程，最大8192个线程块，加速了17.3倍
- **noaudio-blur**：设置1024个线程，最大8192个线程块，加速了23.4倍

## Lab 2

### PART 1

#### Question 1.1: Latency Hiding (5 points)

GK110执行单条指令的延迟为10ns，而在一个时钟周期内，可以执行8条指令（4个warp，每个warp可以执行2条指令）。

时钟频率为1Ghz，即1ns，因此为了掩盖latency，需要执行80条指令。

#### Question 1.2: Thread Divergence (6 points)

线程块的形状为 (32, 32, 1)。

##### (a)

```c
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16)
    foo();
else
    bar();
```

根据线程块的大小，得到`blockSize.x` = `blockSize.y` = 32。

`if`的条件 `idx % 32 < 16`，化简后 `threadIdx.y % 32 < 16`。

而一个warp包含32个线程，即`blockSize.x`，因此同一个warp内的所有线程都会执行同一个函数。

##### (b)

```c
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++)
    result *= pi;
```

所有线程都会执行同一个函数，但是不会出现分支。

#### Question 1.3: Coalesced Memory Access (9 points)

##### (a)
`data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;`

这是合并访问，因为步长为1。

每个warp写入1(32*4/128)个128字节的cache line。

##### (b)
`data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;`

这不是合并访问，因为步长为32。

每个warp写入32(32*32*4/128)个128字节的cache line。

##### (c)
`data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;`

这不是合并访问。虽然步长Wie1，但是有一个偏移量1，这导致同一个warp内的线程访问不同的cache line。

每个warp写入2个128字节的cache line。

#### Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)


```c
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}
```

##### (a)

block的形状为 (32, 32, 1) -> `blockSize.x` = `blockSize.y` = 32。
这意味着同一个warp中的线程有固定的 `threadIdx.y`，而 `threadIdx.x` 在 0 到 31 之间变化。

当一个 warp 中的两个线程访问同一个库中的不同元素时，就会发生 `bank-conflict`。

`i+32*k` -> i = 0,...31 在同一个 warp 中。因此它们访问的是不同的组。-> 没有 `bank-conflict`

`k + 128 * j` -> j 是固定的。它们访问的是同一个组中的相同值。-> 没有 `bank-conflict`

所以在左侧和右侧都不存在 `bank-conflict`。

##### (b)

展开成如下10条指令：
```
(1) x0 = lhs[i + 32 * k]
(2) y0 = rhs[k + 128 * j]
(3) z0 = output[i + 32 * j]
(4) o0 = FMA(x0, y0, z0)
(5) output[i + 32 * j] = o0
(6) x1 = lhs[i + 32 * (k + 1)]
(7) y1 = rhs[(k + 1) + 128 * j]
(8) o1 = output[i + 32 * j]
(9) o1 = FMA(x1, y1, o1)
(10)output[i + 32 * j] = o1
```

##### (c)

依赖关系如下：
```
(1)(2)(3)(4) -> (5)
(5) -> (8)
(6)(7)(8)(9) -> (10)
```

##### (d)

通过临时变量 `x` 和 `y` 的引入，可以消除 (8) 对于 (5) 的依赖。

```
float x = lhs[i + 32 * k] * rhs[k + 128 * j];
float y = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
output[i + 32 * j] += x + y;
```

##### (e)

可以将 `k` 的循环展开从2层变成4层或者8层，以提升整体性能。

### PART 2 - Matrix transpose optimization (65 points)
