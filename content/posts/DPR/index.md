---
title: 解卷积论文1——DPR
date: 2023-11-11T15:41:18+08:00
draft: false
tags: ["解卷积"]
categories: ["科研"]
resources:
- name: "featured-image"
  src: "dpr.png"
- name: "featured-image-preview"
  src: "dpr.png"
lightgallery: true
---

Zhao, B., & Mertz, J. (2023). Resolution enhancement with deblurring by pixel reassignment. Advanced Photonics, 5(6), 066004-066004.

<!--more-->

导师分配给我的毕业课题是解卷积（主要是关于荧光显微镜的），因此我将开始阅读一些经典和较新的论文，这是第一篇。

## 基本信息

{{<admonition type=info title="基本信息" open=true >}}
- **标题**：Resolution enhancement with deblurring by pixel reassignment
- **期刊**：Advanced Photonics
- **作者**：Bingying Zhao, Jerome Mertzb
- **机构**：Boston University
- **日期**：October 2023
- **DOI**：[10.1117/1.AP.5.6.066004](https://www.spiedigitallibrary.org/journals/advanced-photonics/volume-5/issue-06/066004/Resolution-enhancement-with-deblurring-by-pixel-reassignment/10.1117/1.AP.5.6.066004.full?SSO=1)
{{< /admonition >}}

### 关于期刊：Advanced Photonics

{{<figure src="adph.png" title="Advanced Photonics" width="25%">}}

- **领域**：Optics, photonics, and related fields
- **官网**：{{<link "http://advphoton.org" >}}
- **更多信息**：{{<link "https://www.researching.cn/ap/about" >}}

《先进光子学》由中国激光出版社(CLP)、上海光学精密机械研究所(SIOM)和国际光学与光子学学会(SPIE)联合出版，是光学(optics)和光子学(photonics)领域的顶刊。

### 关于作者机构：Boston University Biomicroscopy Lab

{{<figure src="blab.png" title="Biomicroscopy Lab" width="25%">}}

- **研究方向**：开发用于生物医学成像的新显微镜技术
- **PI**：Jerome Mertz
- **主页**：{{<link "https://sites.bu.edu/biomicroscopy/" >}}
- **最近成果**
  - High contrast voltage imaging
  - **Deblurring by pixel reassignment**
  - Imaging at trillion frames per second

都是关于生物医学成像的新型显微镜技术，虽然不是纯做SMLM的，但也可以关注一下。

## 研究背景

## 光学成像与模糊

{{<figure src="blur.png" title="光学成像与模糊(blur)的产生 Credit:Lecture 11, Computational Photography, CMU 15-463" width="50%">}}

光学成像的示意图如上图所示，实际图像等价为PSF和原始图像的卷积后再叠加噪声。

$$
o_k(x)=D[(b(x)*s(\hat{\mathbf{h}}_k(x))) +n_k(x)
$$

在上式中，$b(x)$ 为卷积核，即PSF，$s(\hat{\mathbf{h}}_k(x))$ 为原始图像，$n_k(x)$ 为噪声，$D$ 为下采样(downsampling)操作。

因此所谓的解卷积（deconvolution）就是通过实际图像还原回原始图像。

如果没有PSF的先验信息，则称为盲解卷积（blind deconvolution），否则称为非盲解卷积（non-blind deconvolution）。

{{<figure src="noise.png" title="频域下的信号与噪声 Credit: (Blind) Deconvolution, NSO PHYS 7810" width="50%">}}

模糊是不可避免的，也不存在完美的解卷积，即严格恢复原始图像的算法。

原因在于：首先，PSF的傅里叶变换受到显微镜瞳孔有限尺寸的固有限制，这意味着超过该衍射极限的空间频率相同为零（被截断），因此无法恢复。此外，还有散粒噪声（shot noise）和检测器噪声（detector noise）的存在，这降低了信噪比（SNR），使恢复更加困难。

如上图所示，OTF表现为一个低通滤波器，信号强度随着频率而衰减，而噪声强度则是比较平均的，因而在高频的范围，信号被噪声淹没。

{{<figure src="resolution.png" title="空间分辨率 Credit: Blur & the PSF, Introduction to Bioimage Analysis" width="50%">}}

根据物理光学的理论，存在瑞利判据来描述空间分辨率，即两个点的距离小于瑞利判据时，两个点无法分辨。

如上图所示，随着两个点光源的距离逐渐缩小，两个点的PSF重叠，变得难以分辨（例如图(d)是1个大亮点还是多个小亮点的叠加？）。

在xy平面，$r_{\text{airy}}=\frac{0.61\lambda}{\text{NA}}$，在z轴方向，$r_{\text{airy}}=\frac{1.22\lambda}{\text{NA}^2}$，其中$\lambda$为波长，NA为数值孔径。

对于光学显微镜，$\lambda$ 为数百nm（即可见光波段），而NA在0.1-1.4之间，因此普通光学显微镜的空间分辨率极限在200nm的左右。

### 传统反卷积

回到解卷积，从信号与系统的角度，早在上世纪就有一些被广泛使用的算法被突出，例如Wiener滤波器、Tikhonov正则化、Lucy-Richardson算法等等。

#### Wiener滤波

维纳滤波是基于最大似然的思想，直接对成像公式做傅里叶逆变换。

简化成像公式：

$$
b=k*i+n
$$

直接做傅里叶逆变换 $\mathcal{F}^{-1}$：

$$
i=\mathcal{F}^{-1}(\frac{\mathcal{F}(b)}{\mathcal{F}(k)})
$$

根据上一节的信息，信号在高频部分约为0，无法直接做除法（此时分母为0），因此对分母增加正则化：

$$
i=\mathcal{F}^{-1}(\frac{|\mathcal{F}(k)|^2}{|\mathcal{F}(k)|^2+1/SNR(\omega)}\cdot \frac{\mathcal{F}(b)}{\mathcal{F}(k)})
$$

其中 $SNR(\omega)=\frac{\text{signal variance at }\omega}{\text{noise variance at }\omega}$，即信噪比。

当 $SNR$ 很高时（低频处，信号占主导），$|\mathcal{F}(k)|^2+1/SNR(\omega) \approx \mathcal{F}(k)|^2$，此时退化为简单的逆滤波。

当 $SNR$ 很低时（高频处，噪声占主导），$|\mathcal{F}(k)|^2+1/SNR(\omega) \approx 1/SNR(\omega)$，此时退化为简单的正则化。

这个做法需要已知PSF，且需要知道噪声的功率谱密度（power spectral density, PSD），即 $SNR(\omega)$。

#### Richardson-Lucy算法

最小化负对数似然：

$$
J_1(o)=\sum_{(x,y)}(-i(x,y) \log[(o*k)(x,y)]+(o\*k)(x,y))
$$

迭代计算：

$$
		o_{n+1}(x,y)=o_n(x,y)\odot\{[\frac{b(x,y)}{(h* o_n)(x,y)}] * k^{*}(x,y)\}
$$

增加TV正则项：

$$
		o_{n+1}(x,y)=o_n(x,y)\odot\{[\frac{b(x,y)}{(h* o_n)(x,y)}] * k^{*}(x,y)\}\odot \frac{1}{1+\frac{\partial}{\partial{o}}J_{reg}}
$$

其中

$$
		\frac{\partial}{\partial{o}}J_{reg}=-\lambda_{TV}div(\frac{\nabla o_n(x,y)}{|\nabla o_n(x,y)|})
$$

这个算法在空域下运行，采用迭代的方法。

还有一些其他的算法，在此不再赘述。

### 新型解卷积

在介绍新算法前，先介绍一下艾里斑及其近似。

在 xy 平面，PSF 的中心区域称为艾里斑，占有总能量分布的 84%。我们通常使用高斯函数来近似艾里斑，因为它们的形状非常相似，如下图所示。

{{<figure src="gaussian.png" title="艾里斑与高斯函数对比 Credit: Blur & the PSF, Introduction to Bioimage Analysis" width="50%">}}

#### SRRF

全称为super-resolution radial fluctuations，2016年发表于 {{<link "https://www.nature.com/articles/ncomms12471" "Nature Communications">}}，原理图如下所示。

{{<figure src="srrf.png" title="Credit: Culley et al. (2018)" width="50%">}}

这个算法是基于点扩散函数的对称性，生成径向分布图。结合SOFI的时序波动分析，实现去卷积的效果。

#### MSSR

全程为Mean-Shift Super Resolution，2022年发表于 {{<link "https://www.nature.com/articles/s41467-022-34693-9" "Nature Communications">}}，原理图如下所示。

{{<figure src="mssr.png" title="Credit: Torres-García, Esley, et al. (2022)" width="50%">}}

这个算法是通过mean-shift算法，将艾里斑的中心点从原始图像中分离出来，然后再进行SOFI的时序波动分析，实现去卷积的效果。

{{<admonition type=warning title="坑代填" open=true >}}
接下来的三篇论文为SOFI/SRRF/MSSR，现在写的可能不太准确，后续会更新。
{{</admonition >}}

这两个方法都存在一个非线性的问题，因为都涉及到高阶项的运算，因此最后还需要做一个将光强线性化的处理。

## DPR

### 原理与流程

DPR的基本原理如下图所示：

{{<figure src="dpr.png" title="Credit: Zhao, B., & Mertz, J. (2023)" width="50%">}}

DPR算法的思路比较直接：既然PSF的效果是将中心处的光强分散到周围，那么我们可以通过某种方法将周围像素点的光强重新移动到中心处，从而实现去卷积的效果。

具体的做法是，通过计算每个像素点的梯度的log值，沿着梯度方向将光强重新分配到新的位置，从而实现去卷积的效果，将PSF的宽度缩小。

{{<figure src="algo.png" title="算法流程" width="50%">}}

算法的流程如下：

1. 全局背景减法
2. 归一化到图像中的最大值
3. 重新映射到网格坐标系，网格的宽度大约为PSF的FWHM的八分之一
4. 通过像素重新分配来实现图像的锐化，其中像素的重新分配方向和大小由局部归一化的图像梯度（或等效的log图像梯度）的方向和大小决定，由增益参数缩放（不执行移动距离超过10个像素的）
5. 因为像素通常被重新分配到网格位置，因此它们的像素值按其接近程度加权分布到最近的网格重新分配位置

### 效果

整体效果还是不错的，尤其是对于下图。MSSR和SRRF对尺度较大的物体产生了明显的伪迹（西门子星的内部被掏空，只剩下边缘）。

{{<figure src="compare.png" title="DPR 与 SRRF、MSSR 对比模拟数据西门子星图" width="50%">}}

对于下图，也能明显看出在分岔处，DPR的图像更加清晰。

{{<figure src="real.png" title="DPR 与 SRRF、MSSR 对比实际数据BPAE细胞" width="50%">}}

## 总结

论文中自述的优点有：

- 不太容易产生噪声诱发伪影
- 不需要 PSF 的先验信息
- 保真度较高
- 全空域下进行，无除法运算
- 非迭代算法，无需正则化
- 保留强度信息，无需额外程序保证局部线性

总的来说，DPR算法的思路比较直观，没有复杂的数学推导，但效果不错。

## 参考资料

- [Richard Szeliski, Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)
- [CMU 15-463, 15-663, 15-862 Computational photography Fall 2022](http://graphics.cs.cmu.edu/courses/15-463/2022_fall/)
- [Introduction to Bioimage Analysis](https://bioimagebook.github.io/index.html)
- [Spring 2020 – NSO PHYS 7810: Solar Physics with DKIST](https://nso.edu/students/collage/collage-2020/)