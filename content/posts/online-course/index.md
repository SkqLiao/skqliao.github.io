---
title: CS自学课程汇总
date: 2023-10-11T13:49:09+08:00
draft: false
tags: ["CS"]
categories: ["学习笔记"]
resources:
- name: "featured-image"
  src: "csdiy.png"
- name: "featured-image-preview"
  src: "csdiy.png"
lightgallery: true
---

众所周知，国内大学绝大部分 CS 课程的质量实在堪忧，加之于我本科还是光电的……

本文将持续更新我正在学习的 CS 课程。

<!--more-->

## Haskell MOOC

- **主页**：{{< link "https://haskell.mooc.fi" >}}
- **内容**：Haskell 语言入门，包括类型系统、IO、Monad 等
- **形式**：
  - [ ] 视频
  - [x] 文档
  - [x] 作业(with grader)
- **难度**：Part 1 语言入门很简单；Part 2 中的算子涉及到范畴论，有些难度
- **标签**：函数式编程
- **语言**：Haskell
- **评价**：Haskell 是我接触到的第一门函数式编程语言，相比于 C++等指令式语言有很大差别，整体感觉很有意思，难度也是循序渐进。
- **状态**：已完成

## MIT 6.S081: Operation System Enginnering (2021 Fall)

- **主页**：{{< link "https://pdos.csail.mit.edu/6.828/2021/schedule.html" >}}
- **内容**：在 xv6 上，实现操作系统内核的组件，例如 pagetable、scheduler、file system 等
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
- **难度**：内容很多，但感觉难度不大（因为没有 lab）
- **标签**：深度学习
- **语言**：Python(PyTorch)
- **评价**：收获很大，但对我来说更多的是科普性质的（了解大致原理），没怎么上手写代码。此外，李沐老师每次讲课后还会回答网友的很多问题，这很难得。

## CMU 15-213 Introduction to Computer Systems

- **主页**：{{< link "http://www.cs.cmu.edu/~213/" >}}
- **内容**：
- **形式**：
  - [x] 视频：{{< link href="https://www.bilibili.com/video/BV1iW411d7hd/" content="bilibili" >}}(2015 Fall)
  - [x] 文档：CS-APP 3rd
  - [x] 作业
- **难度**：
- **标签**：操作系统
- **语言**：C、RISC-V
- **评价**：

## 南京大学 操作系统：设计与实现 (2023 春)

- **主页**：{{< link "https://jyywiki.cn/OS/2023/index.html" >}}
- **内容**：
- **形式**：
  - [x] 视频：{{< link href="https://space.bilibili.com/202224425/channel/collectiondetail?sid=1116786&ctype=0" content="bilibili" >}}
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
- **状态**：学习中

## Oxford: Course on CUDA Programming on NVIDIA GPUs, 2023

- **主页**：{{< link "https://people.maths.ox.ac.uk/gilesm/cuda/" >}}
- **内容**：CUDA编程入门
- **形式**：
  - [ ] 视频
  - [x] 文档
  - [x] 作业(no grader): {{< link href="/course-on-cuda-programming-on-nvidia-gpus/" content="my sol" >}}
  - [x] 资料：{{<link href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html" content="NVIDIA CUDA C Programming Guide" >}}
- **难度**：
- **标签**：并行计算
- **语言**：CUDA
- **评价**：
- **状态**：学习中

{{< admonition type=abstract title="课程大纲" open=false >}}

- [x] lecture 1: [An introduction to CUDA](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec1.pdf) (2023/10/11)
- [x] lecture 2: [Different memory and variable types](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec2.pdf) (2023/10/11)
- [x] lecture 3: [Control flow and synchronisation](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec3.pdf) (2023/10/12)
- [x] lecture 4: [Warp shuffles, and reduction / scan operations](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf) (2023/10/12)
- [x] lecture 5: [Libraries and tools](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec5_wes.pdf) (2023/10/13)
- [x] lecture 6: [Multiple GPUs, and odds and ends](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec6_wes.pdf) (2023/10/13)
- [x] lecture 7: [Tackling a new CUDA application](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec7.pdf) (2023/10/18)
- [x] lecture 8: [OP2 "Library" for Unstructured Grids](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec8.pdf) research talk (MG) (2023/10/18)
- [x] lecture 9: [AstroAccelerate](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec9_wes.pdf) research talk (WA) (2023/10/18)
- [x] lecture 10: [Future Directions](https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec10_wes.pdf) (2023/10/18)
- [x] extra research talk: [Use of GPUs for Explicit and Implicit Finite Difference Methods](https://people.maths.ox.ac.uk/gilesm/talks/QuanTech_16.pdf) (2023/10/18)

{{< /admonition >}}

{{< admonition type=info title="作业" open=false >}}

- [x] Practical 1: a trivial "hello world" example (2023/10/11)
- [x] Practical 2: Monte Carlo simulation using NVIDIA's CURAND library for random number generation (2023/10/14)
- [x] Practical 3: 3D Laplace finite difference solver (2023/10/16)
- [x] Practical 4: reduction (2023/10/16)
- [x] Practical 5: using the CUBLAS and CUFFT libraries (2023/10/16)
- [x] Practical 6: revisiting the simple "hello world" example (2023/10/16)
- [x] Practical 7: tri-diagonal equations (2023/10/19)
- [ ] Practical 8: scan operation and recurrence equations
- [ ] Practical 9: pattern matching
- [ ] Practical 10: auto-tuning
- [ ] Practical 11: streams and OpenMP multithreading
- [ ] Practical 12: more on streams and overlapping computation and communication

{{< /admonition >}}

## Caltech CS 179: GPU Programming

- **主页**：{{< link "http://courses.cms.caltech.edu/cs179/" >}}
- **内容**：
- **形式**：
  - [ ] 视频
  - [x] 文档
  - [x] 作业
- **难度**：
- **标签**：并行计算
- **语言**：CUDA
- **评价**：

{{< admonition type=abstract title="课程大纲" open=false >}}

- Week 1 Introduction
  - [x] Lecture 1: Introduction (2023/10/24)
  - [x] Lecture 2: Intro to the SIMD lifestyle and GPU internals (2023/10/24)
  - [x] Lecture 3: Recitation 1 (2023/10/24)

- Week 2 Shared Memory
  - [ ] Lecture 4: 
  - [ ] Lecture 5
  - [ ] Lecture 6

- Week 3 Reductions, FFT
  - [ ] Lecture 7
  - [ ] Lecture 8
  - [ ] Lecture 9

- Week 4 cuBLAS and Graphics
  - [ ] Lecture 10
  - [ ] Lecture 11
  - [ ] Lecture 12

- Week 5 Machine Learning and cuDNN I
  - [ ] Lecture 13
  - [ ] Lecture 14
  - [ ] Lecture 15

- Week 6 Machine Learning and cuDNN II
  - [ ] Lecture 16
  - [ ] Lecture 17
  - [ ] Lecture 18

{{< /admonition >}}

{{< admonition type=info title="作业" open=false >}}

- [x] Lab 1: Introduction to CUDA (2023/10/25)
- [ ] Lab 2:
- [ ] Lab 3:
- [ ] Lab 4:
- [ ] Lab 5:
- [ ] Lab 6:
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
