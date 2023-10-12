---
title: CS自学课程指北
date: 2023-10-11T13:49:09+08:00
draft: false
---

众所周知，国内大学绝大部分CS课程的质量实在堪忧，加之于我本科还是光电的……

本文将持续更新我正在学习的CS课程。

<!--more-->

## Haskell MOOC

- **主页**：{{< link "https://haskell.mooc.fi" >}}
- **内容**：Haskell语言入门，包括类型系统、IO、Monad等
- **形式**：
  - [ ] 视频
  - [x] 文档
  - [x] 作业(with grader)
- **难度**：Part 1语言入门很简单；Part 2中的算子涉及到范畴论，有些难度
- **标签**：函数式编程
- **语言**：Haskell
- **评价**：Haskell是我接触到的第一门函数式编程语言，相比于C++等指令式语言有很大差别，整体感觉很有意思，难度也是循序渐进。

## MIT 6.S081: Operation System Enginnering (2021 Fall)

- **主页**：{{< link "https://pdos.csail.mit.edu/6.828/2021/schedule.html" >}}
- **内容**：在xv6上，实现操作系统内核的组件，例如pagetable、scheduler、file system等
- **形式**：
  - [x] 视频
  - [x] 文档
  - [x] 作业(with grader)
- **难度**：难度不小，需要花费很多时间
- **标签**：操作系统
- **语言**：C
- **评价**：收获很大，在操作系统内核中写代码的逻辑与在应用程序中完全不同，需要考虑很多细节。

## 动手学深度学习 第二版

- **主页**：{{< link "https://zh.d2l.ai" >}}
- **内容**：从零开始深度学习，手把手教你实现深度学习的很多算法
- **形式**：
  - [x] 视频：{{< link href="https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497" content="bilibili" >}}
  - [x] 文档
  - [ ] 作业
- **难度**：内容很多，但感觉难度不大（因为没有lab）
- **标签**：深度学习
- **语言**：Python(PyTorch)
- **评价**：收获很大，但对我来说更多的是科普性质的（了解大致原理），没怎么上手写代码。此外，李沐老师每次讲课后还会回答网友的很多问题，这很难得。

## CMU 15-213 Introduction to Computer Systems
- **主页**：{{< link "http://www.cs.cmu.edu/~213/" >}}
- **内容**：
- **形式**：
  - [x] 视频：{{< link href=htthttps://www.bilibili.com/video/BV1iW411d7hd/" content="bilibili" >}}(2015 Fall)
  - [x] 文档：CS-APP 3rd
  - [x] 作业
- **难度**：
- **标签**：操作系统
- **语言**：C、risc-v汇编
- **评价**：

## 南京大学 操作系统：设计与实现 (2023春)
- **主页**：{{< link "https://jyywiki.cn/OS/2023/index.html" >}}
- **内容**：
- **形式**：
  - [x] 视频：{{< link href=https://space.bilibili.com/202224425/channel/collectiondetail?sid=1116786&ctype=0" content="bilibili" >}}
  - [x] 文档
  - [x] 作业
- **难度**：
- **标签**：操作系统
- **语言**：
- **评价**：

## 國立臺灣大學 Machine Learning (2022 Spring)
- **主页**：{{< link "https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php" >}}
- **内容**：
- **形式**：
  - [x] 视频
  - [x] 文档
  - [x] 作业
- **难度**：
- **标签**：机器学习
- **语言**：
- **评价**：

## Oxford Course on CUDA Programming on NVIDIA GPUs, 2023

- **主页**：{{< link "https://people.maths.ox.ac.uk/gilesm/cuda/" >}}
- **内容**：
- **形式**：
  - [ ] 视频
  - [x] 文档
  - [x] 作业(no grader): {{< link href="/course-on-cuda-programming-on-nvidia-gpus/" content="my sol" >}}
- **难度**：
- **标签**：并行计算
- **语言**：CUDA
- **评价**：

{{< admonition type=abstract title="课程大纲" open=false >}}

- [x] lecture 1: [An introduction to CUDA](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec1.pdf) (2023/10/11)
- [x] lecture 2:  [Different memory and variable types](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec2.pdf) (2023/10/11)
- [ ] lecture 3: [Control flow and synchronisation](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec3.pdf) (2023/10/12)
- [ ] lecture 4: [Warp shuffles, and reduction / scan operations](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf)
- [ ] lecture 5: [Libraries and tools](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec5_wes.pdf)
- [ ] lecture 6: [Multiple GPUs, and odds and ends](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec6_wes.pdf)
- [ ] lecture 7:  [Tackling a new CUDA application](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec7.pdf)
- [ ] lecture 8: [OP2 "Library" for Unstructured Grids](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec8.pdf) research talk (MG)
- [ ] lecture 9: [AstroAccelerate](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec9_wes.pdf) research talk (WA)
- [ ] lecture 10: (https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec10_wes.pdf)
- [ ] extra research talk: [Use of GPUs for Explicit and Implicit Finite Difference Methods](https://people.maths.ox.ac.uk/gilesm/talks/QuanTech_16.pdf)

{{< /admonition >}}

{{< admonition type=info title="作业" open=false >}}

- [x] Practical 1: a trivial "hello world" example (2023/10/11)
- [ ] Practical 2: Monte Carlo simulation using NVIDIA's CURAND library for random number generation
- [ ] Practical 3: 3D Laplace finite difference solver
- [ ] Practical 4: reduction
- [ ] Practical 5: using the CUBLAS and CUFFT libraries
- [ ] Practical 6: revisiting the simple "hello world" example
- [ ] Practical 7: tri-diagonal equations
- [ ] Practical 8: scan operation and recurrence equations
- [ ] Practical 9: pattern matching
- [ ] Practical 10: auto-tuning
- [ ] Practical 11: streams and OpenMP multithreading
- [ ] Practical 12: more on streams and overlapping computation and communication

{{< /admonition >}}

{{< admonition type=note title="补充材料" open=false >}}
{{<link href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html" content="NVIDIA CUDA C Programming Guide" >}}
- [ ] 1. Introduction
- [ ] 2. Programming Model
- [ ] 3. Programming Interface
- [ ] 4. Hardware Implementation
- [ ] 5. Performance Guidelines
- [ ] 6. CUDA-Enabled GPUs
- [ ] 7. C++ Language Extensions
- [ ] 8. Cooperative Groups
- [ ] 9. CUDA Dynamic Parallelism
- [ ] 10. Virtual Memory Management
- [ ] 11. Stream Ordered Memory Allocator
- [ ] 12. Graph Memory Nodes
- [ ] 13. Mathematical Functions
- [ ] 14. C++ Language Support
- [ ] 15. Texture Fetching
- [ ] 16. Compute Capabilities
- [ ] 17. Driver API
- [ ] 18. CUDA Environment Variables
- [ ] 19. Unified Memory Programming
- [ ] 20. Lazy Loading
- [ ] 21. Notices
{{< /admonition >}}


## MIT 6.854/18.415J: Advanced Algorithms (Fall 2021)
- **主页**：{{< link "https://6.5210.csail.mit.edu" >}}
- **内容**：
- **形式**：
  - [x] 视频
  - [x] 文档
  - [x] 作业
- **难度**：
- **标签**：数据结构与算法
- **语言**：
- **评价**：
