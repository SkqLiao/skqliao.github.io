---
title: AtCoder Beginner Contest 自选题解
date: 2023-02-10T11:21:00+08:00
lastmod: 2023-10-08T22:30:00+08:00
draft: false
tags: ["AtCoder", "题解"]
categories: ["算法竞赛"]
resources:
- name: "featured-image"
  src: "atcoder.png"
- name: "featured-image-preview"
  src: "atcoder.png"
lightgallery: true
---

若干ABC的G(F,Ex)题（difficulty>2000）的题解。

<!--more-->

{{< admonition type=tip title="Tip" open=false >}}

鉴于已经退役，算法竞赛只是爱好，因此若无特殊情况，不再更新题解（不过可能会fix typo）。

还记得每天课上都在做题，从ABC212开始，板刷到ABC300(个别G题未补，Ex题仅补了几题)，如今也结束了。

推荐一个看题目difficulty和做题进度的好网站：[AtCoder Problems](https://kenkoooo.com/atcoder#/table/)。

{{< /admonition >}}

## ABC 270 G - Sequence in mod P

**题意：** 序列 $f$ 满足 $f_0=s,f_{i}=a\times f_{i-1}+b(i\geq 1)\pmod p$。给定 $a,b,s,p,g$，求最小的 $n$ 满足 $f_n=g$。

**限制：** $0\leq a,b,s,p,g\leq 10^9$

**链接：** https://atcoder.jp/contests/abc270/tasks/abc270_g

**题解：** BSGS

用 $h_i(x)$ 表示当第 $0$ 项为 $x$ 时，序列第 $i$ 项的值，则 $f_{n=pT-q}=g$ 等价于 $h_{pT}(s)=h_q(g)$。

变换 $f$ 的递推式，设 $h_i(x)=a_i\times x+b_i$，则 $a_i=a_{i-1}\times a,b_{i}=b_{i-1}\times a+b$。

令 $T=\lceil\sqrt{p}\rceil$ ，预处理 $h_q(g):h_1(g),h_2(g),\cdots,h_T(g)$，然后哈希表中依次寻找 $h_{pT}(s):h_T(s),h_{2T}(s),\cdots,h_{T^2}(s)$，最终得到 $n=pT-q$。

已知 $h_{iT}(s)$，则 $h_{(i+1)T}(s)=h_{T}(h_{iT}(s))=a_T\times h_{iT}(s)+b_{T}$，而 $a_T,b_T$ 可以在预处理 $h_{q}(g)$ 时得到，因此 $h$ 的每一项都可以 $O(1)$ 计算。

单组查询的复杂度为 $O(\sqrt{p})$。

**代码：** https://atcoder.jp/contests/abc270/submissions/35189357

## ABC 239 G - Builder Takahashi

**题意：** 给定 $n$ 个点 $m$ 条边的无向连通图，删除点 $i$ 的花费为 $c_i$。可以选择 $2\sim n-1$ 号点中的若干个点删除（并删除与其相连的所有边），使得 $1$ 号点和 $n$ 号点不连通。求最小花费。

**限制：** $n\leq 100$

**链接：** https://atcoder.jp/contests/abc239/tasks/abc239_g

**题解：** 最大流

拆点并建图，连边 $(i, i', c_i)$。对于每条边 $(u,v)$，连边 $(u',v,\infty),(v',u,\infty)$。

最小花费为 $1'\rightarrow n$ 的最大流，被删除的点为最小割上的割边（显然这些边都形如 $i\rightarrow i'$）对应的点。

需要注意的是，最小割上的边并不等价于流量流满（flow=capcity）的边。

复杂度 $O(\text{maxflow}(n, m))$。

**代码：** https://atcoder.jp/contests/abc239/submissions/35192622

## ABC 237 G - Range Sort Query

**题意：** 给定一个长为 $n$ 的排列，有 $q$ 次操作。第 $i$ 次操作为将子区间 $[l_i,r_i]$ 的元素升序/降序排序。求经过 $q$ 次操作后，元素 $x$ 的下标是多少。

**限制：** $n,q\leq 10^5$

**链接：** https://atcoder.jp/contests/abc237/tasks/abc237_g

**题解：** 线段树

区间排序时，只需要知道 $\geq x$（或者 $\leq x$）的元素个数后将左右区间分别覆盖成 $0/1$ ，并更新 $x$ 的位置。

维护一棵区间求和、区间覆盖的线段树，复杂度 $O(q\log{n})$。

**代码：** https://atcoder.jp/contests/abc237/submissions/35194796

## ABC 247 G - Dream Team

**题意：** 有 $n$ 个人，第 $i$ 个人属于 $a_i$ 大学，擅长学科为 $b_i$，能力为 $c_i$。求恰好选择 $1,2,\cdots$ 个人时，在满足任意两个人的大学不同且擅长学科不同的条件下，能力值之和的最大值。

**限制：** $n\leq 3\times 10^4,1\leq a_i,b_i\leq 150$

**链接：** https://atcoder.jp/contests/abc247/tasks/abc247_g

**题解：** 费用流

每所大学和学科都只能被选择一次，这是经典的网络流模型。

$s\rightarrow a_i$，流量为 $1$，费用为 $0$；$b_i\rightarrow t$，流量为 $1$ ，费用为 $0$；$a_i\rightarrow b_i$，流量为 $1$，费用为 $-c_i$。

跑最小费用最大流。每增广一次，流量增加 $1$，即多选一个人，累计的总费用即为此时的能力值之和。

注意要用增广的方式跑费用流，而不是每次从 $s$ 增加一条边容量为 $1$ 的边再重新跑pushRelabel。

**代码：** https://atcoder.jp/contests/abc247/submissions/35195856

## ABC 238 G - Cubic?

**题意：** 给定长为 $n$ 的序列 $a$，有 $q$ 个询问。第 $i$ 次询问给定 $l_i,r_i$，查询 $\prod\limits_{j=l_i}^{r_i}{a_j}$ 是否是立方数。

**限制：** $n,q\leq 2\times 10^5,1\leq a_i\leq 10^6$

**链接：** https://atcoder.jp/contests/abc238/tasks/abc238_g

**题解1：** 莫队

对 $a_i$ 做质因数分解，不难发现至多 $8$ 个质因数（$2\times 3\times 5\times 7\times 11\times 13\times 17\times 19=9,699,690>1,000,000$）。而立方数只需要满足每个质因子的幂指数都为 $3$ 的倍数。因此可以用莫队维护区间中每个质因子的幂指数。复杂度 $O(q\sqrt{n})$。

其实莫队的部分跑的并不慢，但如果在预处理的时候暴力分解质因数，那就寄了，需要用埃氏筛预处理每个数的最小质因数，即可实现 $O(\log{a_i})$ 分解。

**题解2：** 哈希

分析同上，我们需要维护的是每个质因数的幂指数。不难发现不同质因子之间互不干扰。因此可以通过哈希的方式维护每个前缀区间的 $\sum\limits_{p\in \text{prime}}{p\times (cnt[p]\bmod 3)}$ ，其中 $cnt[p]$ 为质因子 $p$ 的出现次数。若两前缀值相等，则说明该区间中所有质因子的幂指数都是 $3$ 的倍数。复杂度 $O(n\log{a_i})$。

由于本方法复杂度低，因此即使你暴力分解质因数，代码跑的也不慢。。

**代码1：** https://atcoder.jp/contests/abc238/submissions/35207997

**代码2：** https://atcoder.jp/contests/abc238/submissions/35208067

## ABC 267 G - Increasing K Times

**题意：** 给定长为 $n$ 的序列 $a$，求有多少个 $1\sim n$ 的排列 $p$ ，满足恰好存在 $k$ 个 $i$ 满足 $a_{p_i}<a_{p_{i+1}}$。答案对 $998244353$ 取模。

**限制：** $n\leq 5000$

**链接：** https://atcoder.jp/contests/abc267/tasks/abc267_g

**题解：** DP

问题等价于求 $a$ 有多少种排列方式 $b$，满足存在 $k$ 个「正序对」（$b_i<b_{i+1}$）。考虑将 $a$ 升序排序后，逐个插入空序列中，则每插入一个数，新序列的「正序对」个数要么不变，要么增加 $1$。

将 $a_i$ 插入在 $b_j,b_{j+1}$ 之间（令 $b_{-1}=b_{ \vert b\vert +1 }=0$），根据大小关系有两种情况：
- $a_i>b_j,b_j\geq b_{j+1}$：个数加一
- 其他：个数不变

记 $f(i,j)$  为前 $i$ 个数，组成 $j$ 个「正序对」的方案数。（记 $cnt[a_i]$ 为 $a_i$ 的当前出现次数）
- $f(i,j)\rightarrow f(i+1,j)$：插在与 $a_i$ 相等的数 / $b_j<b_{j+1}$ 之间，共 $cnt[a_i]+j+1$ 种方式
- $f(i,j)\rightarrow f(i+1,j+1)$：共 $i-j-cnt[a_i]$ 种方式

答案为 $f(n,k)$，复杂度 $O(nk)$。

**代码：** https://atcoder.jp/contests/abc267/submissions/35214949

## ABC 256 G - Black and White Stones

**题意：** 将若干个黑白石子摆成每条边上有 $d+1$ 个石子的 $n$ 边形，要求每条边上的白色石子个数相同，求方案数。答案对 $998244353$ 取模。

**限制：** $n\leq 10^{14}, d\leq 10^4$

**链接：** https://atcoder.jp/contests/abc256/tasks/abc256_g

**题解：** 矩阵快速幂

枚举每条边的白色石子数 $m$，设 $f^m_{0/1}(i,0/1)$，表示已经放置了 $i$ 条边，其中第 $1$ 条边的首和第 $i$ 边的尾分别为黑/白石子的方案数，则答案为 $\sum\limits_{m=0}^{d+1}{f^m_1(n,1)+f^m_0(n,0)}$。

则转移方程 $f^m_{\*}(i)=a^m\times f^m_{\*}(i-1)$ 为：

$$ \begin{bmatrix} f(i, 0)\\ f(i, 1) \end{bmatrix} = \begin{bmatrix} {d-1\choose m-2}\ {d-1\choose m-1}\\ {d-1\choose m-1}\ {d-1\choose m} \end{bmatrix}\times \begin{bmatrix} f(i-1, 0)\\ f(i-1, 1) \end{bmatrix} $$

这个方程可以用矩阵快速幂转移，总复杂度 $O(d\times \log{n})$。

**代码：** https://atcoder.jp/contests/abc256/submissions/35216315

## ABC 246 G - Game on Tree 3

**题意：** 给定一棵树，除根节点外的任意点 $i$ 都有权值 $a_i$。Alice和Bob轮流操作，从根节点出发。在每个回合中，Alice首先选择任意一个点，将其权值置为 $0$，然后Bob从当前点移动到它的任意一个子节点中，并可以选择立刻结束游戏（若到达叶子节点，则游戏自动结束）。游戏得分是结束游戏前他所在点的权值，Alice想要最小化，Bob想要最大化。如果双方均采取最优策略，求最终得分。

**限制：** $n\leq 2\times 10^5,1\leq a_i\leq 10^9$

**链接：** https://atcoder.jp/contests/abc246/tasks/abc246_g

**题解1：** 二分答案+树形DP

首先二分答案 $x$，将权值小于 $x$ 的点的权值 $b_i$ 置为 $0$，否则置为 $1$。问题变成Bob能否达到一个权值为 $1$ 的点。

设 $f(x)$ 表示如果下一步走到 $x$ 点，在此之前，Alice需要至少多少次置 $0$ 才能阻止未来Bob到达权值为 $1$ 的点。转移方程为 $f(x)=\max(-1+\sum\limits_{y\in son[y]}{f(v)},0)+b_x$，最后只需要判断 $f(1)$ 是否为 $0$。

复杂度 $O(n\log{a_i})$。

**题解2：** 线段树

同样是自底向上考虑。贪心地想，当前在 $x$ 点，Alice下一回合会置 $0$ 的点应该是 $x$ 子树中（不包括 $x$）剩余点中的最大值。因此可以用线段树维护子树最大值，自底向上转移。

复杂度 $O(n\log{n})$。

**代码1：** https://atcoder.jp/contests/abc246/submissions/35227917

**代码2：** https://atcoder.jp/contests/abc246/submissions/me

## ABC 234 G - Divide a Sequence

**题意：** 给定长为 $n$ 的序列 $a$，共有 $2^n$ 种方式将其划分成长度为 $m=1,2,\cdots,n$ 的非空连续子序列 $b_1,b_2,\cdots,b_m$。对于任意划分，定义其权值为 $\prod{\max(b_i)-\min(b_i)}$。求这 $2^n$ 个划分的权值之和，答案对 $998244353$ 取模。

**限制：** $n\leq 3\times 10^5$

**链接：** https://atcoder.jp/contests/abc234/tasks/abc234_g

**题解：** 单调栈

$f(i)$ 表示前 $i$ 个数的所有划分的权值之和，则 $f(i)=\sum\limits_{j<i}{f(j)\times (\max\limits_{k=j+1}^{i}{a_k}-\min\limits_{k=j+1}^{i}{a_k})}$。

将后式的 $\max$ 和 $\min$ 拆开维护，显然 $[1,i-1]$ 可以划分出若干段连续区间，其区间 $\max$ / $\min$ 是相等的，且区间的从左到右 $\max/\min$ 值是递减/递增的。

因此可以用单调栈维护这些 $\max/\min$ 的区间 $f$ 值之和，弹出栈顶的同时计算出它的贡献增量。复杂度为 $O(n)$。

**代码：** https://atcoder.jp/contests/abc234/submissions/35230774

## ABC 223 G - Vertex Deletion

**题意：** 给定一棵树，求有多少个点满足在删除它之后，最大匹配数不变。

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc223/tasks/abc223_g

**题解：** 换根DP

求树的最大匹配有 $O(n)$ 的树形DP，即 $f[x][0/1]$ 表示 $x$ 子树中的最大匹配数，且 $x$ 是否与它父亲连边。

转移方程为 $f[x][1]=\sum\limits_{y\in son[x]}{f[y][0]},f[x][0]=\sum\limits_{y\in son[x]}{f[y][0]}+\max\limits_{y\in son[x]}{f[y][1]+1-f[y][0]}$。

同时记录 $x$ 的子节点中，$f[y][1]+1-f[y][0]$ 的最大值和次大值来源，记为 $\text{from}[x][0/1]$。

换根DP， $x\rightarrow y$ 时维护 $f[x][\*],f[y][\*],\text{from}[x][\*],\text{from}[y][\*]$ 的变化值即可，其他点的值不变。

复杂度 $O(n)$。

**代码：** https://atcoder.jp/contests/abc223/submissions/35242470

## ABC 230 G - GCD Permutation

**题意：** 给定一个长为 $n$ 的排列 $p$，求有多少个数对 $(i,j)(i\leq j)$ 满足 $\gcd(i,j)\not=1$ 且 $\gcd(p_i,p_j)\not=1$。

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc230/tasks/abc230_g

**题解：** 莫比乌斯反演

设 $f(a,b;i,j)$ 表示[$i,j$ 是 $a$ 的倍数，且 $p_i,p_j$ 是 $b$ 的倍数]。

由 $\sum\limits_{d\mid n}\tilde\mu(d)=[n\geq 2]$，枚举位置 $i,j$ ，并枚举公因数 $a,b$，则答案为 $\sum\limits_{1\leq i\leq j\leq n}\sum\limits_{a=1}^{n}\sum\limits_{b=1}^{n}{\tilde\mu(a)\tilde\mu(b)f(a,b;i,j)}$，其中 $\mu(a)$ 为莫比乌斯函数。

记 $h(a,b)$ 为满足 $i\mid a$ 且 $p_i\mid b$ 的 $i$ 的个数，则 $\sum\limits_{1\leq i\leq j\leq n}{f(a,b;i,j)}=\frac{h(a,b)(h(a,b)+1)}{2}$，答案化简为 $\sum\limits_{a=1}^{n}\sum\limits_{b=1}^{n}\tilde\mu(a)\tilde\mu(b)\frac{h(a,b)(h(a,b)+1)}{2}$。

先枚举 $a$，无需枚举 $b$。此时满足 $i\mid a$ 的数为 $p_{a},p_{2a},\cdots,p_{ka}$，则 $b$ 需要是他们的因数之一且 $\tilde\mu(b)\not=0$。因此对于 $p_{ia}$ ，遍历它的所有 $\tilde\mu(d)\not=0$ 的因子 $d$，并统计出现次数。

枚举的数总个数 $\{p_{ia}\}=\sum\limits_{i=1}^{n}{\lfloor\frac{n}{i}\rfloor}=O(n\log{n})$ ，而 $\leq 2\times 10^5$ 的数的质因子个数不会超过 $7$ 个，因而 $\tilde\mu\not=0$ 的因数个数不超过 $2^7$ 个，总复杂度 $O(63n\log{n})$。

**代码：** https://atcoder.jp/contests/abc230/submissions/35244167

## ABC 236 G - Good Vertices

**题意：** $n$ 个点的有向图，在第 $0$ 秒时没有边。在第 $i(1\leq i\leq t)$ 秒，加入有向边 $u_i\rightarrow v_i$。对于任意点 $t$，求长度恰好为 $L$ 条边的路径 $1\to t$ 的最早出现时间。

**限制：** $n\leq 100, t\leq n^2,L\leq 10^9$

**链接：** https://atcoder.jp/contests/abc236/tasks/abc236_g

**题解：** 矩阵快速幂

设 $f(i,j)$ 表示到达经过恰好 $i$ 条边到达 $j$ 点的所有路径中编号最大的边的最小值，答案为 $f(L,1),\cdots,f(L,n)$。

转移方程：$f(i,j)=\min\limits_{1\leq j\leq n}\{\max(f(i-1,j),W(j,i))\}$，其中 $W(j,i)$ 为边 $j\rightarrow i$ 出现的时间，若不存在置为 $\infty$。

初值 $f(0,1)=0,f(0,i)=\infty(2\leq i\leq n)$。

将 $\max$ 定义为 $+$，$\min$ 定义为 $\times$，则上式可以写成矩阵乘法的形式 $f_{n\times 1}(i)=W_{n\times n}\times f_{n\times 1}(i-1)$。

由于 $\max,\min$ 满足分配律、交换律和结合律，因此可以改变运算顺序，对这个式子做矩阵快速幂，得到 $f(L,i)=W^L(i,1)$。

复杂度 $O(n^3\log{L})$。

**代码：** https://atcoder.jp/contests/abc236/submissions/35326063

## ABC 249 G - Xor Cards

**题意：** 有 $n$ 张卡片，第 $i$ 张卡片正面写着 $a_i$，背面写着 $b_i$。现在选择若干张卡片，求在满足正面的异或和不超过 $k$ 的条件下，背面的异或和的最大值。

**限制：** $n\leq 1000,0\leq a_i,b_i,k\leq 2^{30}$

**链接：** https://atcoder.jp/contests/abc249/tasks/abc249_g

**题解：** 线性基

根据线性基定理，$n$ 个数的任意子集的异或和只需要 $O(\log{V})$ 个数的线性基即可表示。

对于两张卡片 $(a,b),(c,d)$，它们等价于 $(a\oplus c,b\oplus d),(c,d)$，因此可以通过这样的方式，一边构造 $a_i$ 的线性基 $la$，并把所有能用线性基表示的卡片 $(a_i,b_i)$ 转换成 $(0,b_i')$。

此时剩下若干个 $(0,b_i')$ 和一个线性基 $la$，前者可以再构造出一个对 $k$ 的限制无影响的线性基 $lb$。

此时考虑如何满足 $k$ 的限制。

枚举 $i$，用 $la$ 构造出前 $i-1$ 位与 $k$ 相同，且当前位为 $0$ 而 $k_{(i)}=1$ 的状态，此时任意取后面的位都满足限制。那么就可以将后面位的 $b_i$ 插入 $lb$ 中，并查询异或最大值。

复杂度 $O(n\log{V}+\log^3{V})$。

**代码：** https://atcoder.jp/contests/abc249/submissions/35388300

## ABC 248 G - GCD cost on the tree

**题意：** 给定一棵树，点 $i$ 有点权 $a_i$。定义 $f(u,v)$ 为 $u,v$ 之间的最短路径上的点数与最短路径上所有点权值的 $\gcd$ 的乘积。求 $\sum\limits_{i=1}^{n}\sum\limits_{j=i+1}^{n}{f(i,j)}$。

**限制：** $1\leq n,a_i\leq 10^5,$

**链接：** https://atcoder.jp/contests/abc248/tasks/abc248_g

**题解：** 树形DP

应该算是个暴力，也不知道复杂度对不对，不过跑的飞快。

维护 $f(i)=\{(x,y)\}$ ，表示 $i$ 的子树中的所有点，从 $i$ 出发，路径 $\gcd=x$ 的所有点的深度之和为 $y$。

那么合并 $f(i)$ 和其子节点 $f(j)$ 时，只需要分别枚举 $(x_1,y_1)\in f(i)$ 和 $(x_2,y_2)\in f(j)$，贡献为 $[y_1\times x_2+x_1\times y_2-(2\times dep(i) - 1)\times (y_1\times y_2)] \times \gcd(x_1,x_2)$。

注意到，虽然看起来 $x$ 的取值有很多，但是实际上至多只有 $d(a_i)$ 个。而根据经典老图，$\max\{d(10^5)\}=128$，因此复杂度上界是 $O(128^2\times n)$，但实际上远远跑不满。

{{< image src="https://raw.githubusercontent.com/SkqLiiiao/image/main/202210041448144.jpeg" caption="UOJ经典老图" >}}

**代码：** https://atcoder.jp/contests/abc248/submissions/35390296

## ABC 225 G - X

**题意：** 给定一个 $n\times m$ 的网格，每个格子有权值 $w(i,j)$。现在可以选择若干个格子画"X"，即两条对角线。最后的得分为所有画"X"的格子的权值之和- $C\times$ 需要画的对角线数量（每次画对角线可以穿越多个格子）。要求每个格子要么画"X"，要么什么都没画。求得分的最大值。

**限制：** $n,m\leq 100$

**链接：** https://atcoder.jp/contests/abc225/tasks/abc225_g

**题解：** 最大流

现在的得分是两部分的差，不太好算。先假设所有格子都被画"X"获得 $\sum{w(i,j)}$ 的分数，再减去付出代价的最小值，如此得到的结果应与答案相同。

先考虑右下方向的对角线，它的代价应该是 $C\times$ 【 $(i,j)$ 画"X"，但是 $(i-1,j-1)$ 不画的数量】，同理左下方向的。

因此构图方法如下：

- $s\rightarrow (i,j)$，流量为 $w(i,j)$
- $(i,j)\rightarrow (i-1,j-1)$，流量为 $c$（存在 $(i-1,j-1)$）
- $(i,j)\rightarrow (i-1,j+1)$，流量为 $c$（存在 $(i-1,j+1)$）
- $(0,j)/(i,0)\rightarrow t$，流量为 $c$
- $(0,j)/(i,m)\rightarrow t$，流量为 $c$ 

答案为 $\sum{w(i,j)}-\text{maxflow}(s,t)$。

**代码：** https://atcoder.jp/contests/abc225/submissions/35391979

## ABC 271 G - Access Counter

**题意：** 给定长为 $24$ 的字符串，$s[i]=a$ 表示每天 $i$ 点钟Alice有 $x$ 的概率访问网页，$s[i]=b$ 表示每天 $i$ 点钟Bob有 $y$ 的概率访问网页。从某天的 $0$ 点开始游戏，求第 $n$ 次访问网页恰好是Alice完成的概率。

**限制：** $n\leq 10^{18}$

**链接：** https://atcoder.jp/contests/abc271/submissions/35417575

**题解：** 矩阵快速幂

设 $f(i,j,k)$ 表示第 $2^i$ 次访问网页在 $k$ 点钟，上一次访问在 $j$ 点钟的概率，转移方程为 $f(i,j,k)=\sum\limits_{l=0}^{23}{f(i-1,j,l)\times f(i-1,l,k)}$。

计算初值，假设第 $1$ 次访问网页是在 $i$ 点钟，则它需要经过 $n(n\geq 0)$天的失败，以及 $[0,i-1]$ 点钟的失败和 $i$ 点钟的成功，概率为 $g(i)=\sum\limits_{n=0}^{\infty}{q^n\times \prod\limits_{j=0}^{i-1}{(1-p_j)}\times p_i}$，其中 $q=\prod{(1-p_i)}$，$p_i$ 为 $i$ 点钟访问网页的概率。

根据等比数列求和公式，$g(i)=\frac{1}{1-q}\times\prod\limits_{j=0}^{i-1}{(1-p_j)}\times p_i$。

计算出 $g(i)$ 后，通过枚举第 $1$ 次和第 $2$ 次访问时间可以得到初值 $f(1,\*,\*)$。

不难发现转移方程矩阵乘法的形式，因此可以通过矩阵快速幂求第 $n$ 次访问的概率 $h(\*,\*)$，答案为 $\sum h(23,i)$，其中 $i$ 满足 $s[i]=a$。

复杂度 $O(24^3\log{n})$。

**代码：** https://atcoder.jp/contests/abc271/tasks/abc271_g

## ABC 235 G - Gardens

**题意：** 有三种石子，分别有 $a,b,c$ 个放置在 $n$ 个有标号的盒子中。要求每个盒子至少放置一个石子，每个盒子中至多一个同种石子，而且不必放完所有石子。求方案数。

**限制：** $n \leq 5\times 10^6$

**链接：** https://atcoder.jp/contests/abc235/tasks/abc235_g

**题解：** 反演

先考虑至少 $i$ 个盒子为空的方案数，$\binom{n}{i}\sum\limits_{x=0}^{\min(a,n-i)}\binom{a}{i}\sum\limits_{y=0}^{\min(b,n-i)}\binom{b}{y}\sum\limits_{z=0}^{\min(c,n-i)}\binom{c}{z}$。

则答案为恰好 $0$ 个盒子为空的方案数，容斥一下，得到 $\sum\limits_{i=0}^{n}(-1)^{i}\binom{n}{i}\sum\limits_{x=0}^{\min(a,n-i)}\binom{a}{i}\sum\limits_{y=0}^{\min(b,n-i)}\binom{b}{y}\sum\limits_{z=0}^{\min(c,n-i)}\binom{c}{z}$

用 $f_M(N)$ 表示 $\sum\limits_{i=0}^{\min(N,M)}\binom{N}{i}$，则答案化简为 $\sum\limits_{i=0}^{n}(-1)^{n-i}\binom{n}{i}f_a(i)f_b(i)f_c(i)$。

现在考虑如何从 $f_M(N)$ 快速计算出到 $f_M(N+1)$。

如果 $N+1\leq M$，根据二项式定理， $f_M(N)=\sum\limits_{i=0}^{N}{\binom{N}{i}}=2^N$，$f_M(N+1)=2^{N+1}$，得 $f_M(N+1)=2f_M(N)$。

否则 $f_M(N)=\sum\limits_{i=0}^{M}\binom{N}{i}$，$f_M(N+1)=\sum\limits_{i=0}^{M}\binom{N+1}{i}$。

根据 $\binom{N}{M}=\binom{N-1}{M}+\binom{N-1}{M-1}$，则 $2\sum\limits_{i=0}^{M}\binom{N}{i}=\sum\limits_{i=0}^{M}\binom{N+1}{i}-\binom{N}{M}$。

也可以通过下图考虑它的意义：

{{< image src="https://img.atcoder.jp/ghi/d93c643497867d310c6255fb673d9682.png" caption="来自官方题解" >}}

从左下角出发，每次向右/上移动一格，则到达黄色点的方案数之和为 $f_M(N)$，到达蓝色点的方案数为 $f_M(N+1)$。

不难发现除了右下角的黄点无法向右移动以外，其他点都有向上/右移动两种方案，因此 $f_M(N+1)=2f_M(N)-\binom{N}{M}$。

如此每次转移就是 $O(1)$ 的了，总复杂度 $O(n)$。

**代码：** https://atcoder.jp/contests/abc235/submissions/35426630

## ABC 232 G - Modulo Shortest Path

**题意：** 有 $n$ 个点的有向完全图，每个点有权值 $a_i$ 和 $b_i$。有向边 $i\rightarrow j$ 的边权为 $(a_i+b_j)\bmod m$ ，求 $1\to n$ 的最短路。

**限制：** $n\leq 2\times 10^5, 0\leq a_i,b_i\leq m,2\leq m\leq 10^9$

**链接：** https://atcoder.jp/contests/abc232/tasks/abc232_g

**题解：** 建图+最短路

显然需要把边的数量降下来才能用dijkstra求最短路，因此考虑构建与原图等价的新图。

对于权值 $0\sim m-1$ 分别建一个新点，$[i]\rightarrow [(i+1)\bmod m]$ 边权为 $1$，可以想象将这 $m$ 个点排列成一个顺指针的圆环。

对于原图上的 $n$ 个点，$i\rightarrow j$ 的边权为 $(a_i+b_j)\bmod m$，考虑用这个圆环上的边替代，则可以用 $i\rightarrow [m-a_i]$ 连边权为 $0$ 的边，再连一条 $[b_j]\rightarrow j$ 边权为 $0$ 的边。此时 $i\rightarrow j$ 就可以沿着圆环走，路上的边权和仍然为 $(a_i+b_j)\bmod m$。

显然不是圆环上的所有点都有用，那些除圆环外没有入度的点可以被删除合并，这样圆环上至多留下 $2n$ 个点。

现在一共 $3n$ 个点，$4n$ 条边，可以用dijkstra $O(n\log{n})$ 求出 $1\to n$ 的最短路。

**代码：** https://atcoder.jp/contests/abc232/submissions/35428237

## ABC 228 G - Digits on Grid

**题意：** 在一个 $h\times w$ 的网格图上，每个格子上有 $0\sim 9$ 的数字。初始状态将棋子放在任意格子上。在每一轮操作中，首先将棋子移动到与当前状态同属一行的任意格子上（可以不动），并记录移动后所在格子上的数字；然后将格子移动到与当前状态（经过上次同属一行的移动后）同属一列的任意格子上（可以不动），并记录移动后所在格子的数字。经过如此 $n$ 轮操作，会得到一个长为 $2n$ 的数字串，求有多少种不同串，答案对 $998244353$ 取模。

**限制：** $h,w\leq 10, n\leq 300$

**链接：** https://atcoder.jp/contests/abc228/tasks/abc228_g

**题解：** 记忆化搜索

跟题解状压DP做法不太一样，但可能本质上是相同的。

首先，注意到如果上一轮操作后棋子位于同一行，那么接下来操作的所有状态都是相同的。

其次，每一轮字符串的长度会增加 $2$，如果这两位不同，那么无论后面如何操作，得到的字符串肯定是不同的。

第三，操作后所得到的相同字符串可能会终止在不同格子上，他们的所在行构成一个集合。

因此我们可以维护 $f_n(x)$，表示经过 $n$ 轮操作后，从行状态为 $x$ （01串，$[i]=1$ 表示初始状态在第 $i$ 行）出发的不同字符串的数量，答案为 $\sum\limits_{i=1}^{h}{f_n(i)}$。

而从一个行集合出发，经过一轮操作后得到的不同字符串至多只有 $10^2$ 个，且对应状态的行集合也是确定的，状态数都很少。

我们可以预处理出，初始状态行集合为 $x$（01串），经过一轮操作得到新字符串为 $ab$ 的终止状态的行集合（01串）$f[x][a][b]$。

01串用bitset<10>存储，计算出以每一行作为初始状态后，类似状压DP的方式将 $x$ 从小到大更新，复杂度为 $O(2^{10}\times 10\times 10\times \frac{10}{64})$。

预处理完成后，直接DFS作记忆化搜索，即可算出答案。

由于 $f_n(x)$ 的总状态数只有 $n\times 2^{h}\leq 3\times 10^5$，因此跑的飞快。写到这儿才发现，大概题解的状压也是这个意思吧。。

**代码：** https://atcoder.jp/contests/abc228/submissions/35430596

## ABC 272 **G - Yet Another mod** M

**题意：** 给定长为 $n$ 的序列 $a$，求 $m(3\leq m\leq 10^9)$，满足 $b_i=a_i\bmod m$ 中存在一个出现次数超过 $\lfloor\frac{n}{2}\rfloor$ 的众数。

**限制：** $3\leq n\leq 5000，1\leq a_i\leq 10^9$

**链接：** https://atcoder.jp/contests/abc272/tasks/abc272_g

**题解：** 同余

观察1：$x\equiv y\pmod m$ 等价于 $m \mid(x-y)$。

观察2：若合数 $m_0$ 满足条件，那么它的所有因数都满足条件。

由于出现此时要超过一半，因此总会存在 $i$ 满足 $a_i$ 和 $a_{i+1}$ 或者 $a_i$ 和 $a_{i+2}$ 同时出现。

因此一个思路是，依次判断 $\vert a_{i+1}-a_i\vert$ 和 $\vert a_{i+2}-a_i\vert$ 的所有因数是否成立，又根据观察2，只需要枚举其质因子。

复杂度 $O(n^2\log{a_i})$。

**代码：** https://atcoder.jp/contests/abc272/submissions/35514651

## ABC 231 G - Balls in Boxes

**题意：** 有 $n$ 个盒子，第 $i$ 个盒子初始状态有 $a_i$ 个球。每轮操作时，随机将一个新球放入一个盒子中。设 $m$ 轮操作后，第 $i$ 个盒子中放了 $b_i$ 个球，求 $\prod{(a_i+b_i)}$ 的期望值。

**限制：** $n\leq 1000,m\leq 10^9$

**链接：** https://atcoder.jp/contests/abc231/tasks/abc231_g

**题解：** 期望

$O(n\log^2{n})$ 的FFT不会，这是 $O(n^2)$ 的做法。。。

显然总共有 $n^m$ 种不同情况，因此答案为所有情况的 $\prod{(a_i+b_i)}$ 之和除以 $n^m$。

先不考虑 $a_i$，不妨将其全部视为 $0$，求 $\sum\prod{b_i}$ 。

令 $a_{ij}$ 表示【第 $i$ 个盒子是否放进了第 $j$ 个球】，值为 $0$ 或 $1$，则 $\prod{b_i}=(a_{11}+\cdots+a_{1m})(a_{21}+\cdots+a_{2m})\cdots(a_{n1}+\cdots+a_{nm})$，共 $m^n$ 项，每一项为 $0$ 或 $1$。

考虑任意一项 $a_{1t_1}a_{2t_2}\cdots a_{nt_n}=1$，此时需要满足 $\forall i,a_{it_i}=1$，且 $\forall i<j,t_i\not=t_j$（一个球不能同时放入两个盒子中），因此 $\{t_i\}$ 共有 $m!(m-1)!\cdots(m-n+1)!=\frac{m!}{(m-n)!}$。

对于一组 $\{t_1,\cdots,t_n\}$，先放入每个袋子放入对应的球，剩余 $m-n$ 个球可以随意放，有 $n^{m-n}$ 种情况。

因此 $\sum\prod{b_i}=\frac{m!n^{n-m}}{(m-n)!}$。

现在再考虑 $a_i\not=0$ 的情况。

将 $\prod{(a_i+b_i)}$ 拆开，每一项应该包括 $x$ 个 $a_i$ 和 $n-x$ 项 $b_i$，前者的乘积记为 $f(x)$，后者的乘积记为 $g(n-x)$。

$f(x)$ 可以通过递推求出。

令 $w(i,j)$ 表示从 $a_1,\cdots,a_i$ 中选择 $j$ 个（$a_{b_1},\cdots,a_{b_j}$）的 $\prod\limits{a_{b_k}}$ 之和，则 $w(i,j)=w(i-1,j)+a_i\times w(i-1,j-1)$。

则 $f(x)=w(n,x)$。

而 $g(n-x)$ 可以同理 $a_i=0$ 时求 $\sum\prod{b_i}$ 的方式， $g(x)=\frac{m!n^{m-x}}{(m-x)!}$。

答案为 $\sum\limits_{i=0}^{n}{f(x)\times g(n-x)}$。

复杂度瓶颈在求 $f(x)$，因此复杂度为 $O(n^2)$。

**代码：** https://atcoder.jp/contests/abc231/submissions/35618574

## ABC 264 G - String Fair

**题意：** 给定 $n$ 个长度不超过 $3$ 的字符串 $t_i$ 以及对应分数 $p_i$（其他串分数为 $0$）。求所有仅包含小写字母的非空字符串 $s$ 的价值最大值（具体值或者无限大）。字符串的价值定义为其所有子串的分数之和。

**限制：** $n\leq 18278$

**链接：** https://atcoder.jp/contests/abc264/tasks/abc264_g

**题解：** 最长路

设字符串 $s$ 的价值为 $f(s)$，字符串 $t$ 的分数为 $g(t)$。

假设当前字符串为 $s=s_1s_2,\cdots,s_m$，在末尾插入字符 $c$，得到 $s'$，则 $f(s')=f(s)+g(s_{m-1}s_{m}c)+g(s_mc)+g(c)$。

发现只需要记录末尾两位字符即可表示当前字符串的状态，因此可以维护 $26\times 26$ 个点的有向图，求最长路，用Bellman-Ford判断是否存在正环。

复杂度 $O(26^5)$。

**代码：** https://atcoder.jp/contests/abc264/submissions/35787051

## ABC 263 G - Erasing Prime Pairs

**题意：** 给定 $n$ 个正整数及其出现次数 $(a_i,b_i)$。每次操作可以选择两个数，若它们的和是质数，则将这两个数删去。求最多操作多少次。

**限制：** $n\leq 100, a_i\leq 10^7,b_i\leq 10^9$

**链接：** https://atcoder.jp/contests/abc263/tasks/abc263_g

**题解：** 二分图匹配

除了 $2$ 以外的所有质数都是奇数，只能通过奇数+偶数的方式凑出。

先不考虑 $1$。将其他数分成奇数和偶数两部分，将和是质数的两数之间连边，这是个二分图。答案为二分图最大匹配数。

问题在于 $1$ 可以和自己匹配。

考虑枚举 $(1,1)$ 匹配的次数 $x$，剩余 $1$ 和其他数做二分图最大匹配数记为 $f(x)$，则总答案为 $x+f(x)$。

现在观察 $f(x)$，不难发现它性质特殊，是个凹函数，可以三分求出极值。

复杂度 $O(\log{b_i}\text{maxflow}(n,n^2))$

**代码：** https://atcoder.jp/contests/abc263/submissions/35791530

## ABC 252 G - Pre-Order

**题意：** 对于根为 $1$，有 $n$ 个点的树，DFS时要求优先遍历标号较小的儿子对应的子树，并记录DFS序。现给定DFS序，求它对应着多少棵不同的树。

**限制：** $n\leq 500$

**链接：** https://atcoder.jp/contests/abc252/tasks/abc252_g

**题解1：** 区间DP

任意子树的所有点在DFS序上一定组成一个连续子区间。

设 $f(l,r)$ 表示子区间 $[l,r]$ 对应的树的个数，则答案为 $f(1,n)$。

现在考虑对于当前DFS序列的区间 $[l,r]$，如何拆成若干个不相交非空子区间 $[l_i,r_i]$。根据题目要求，子树根的标号在DFS序中从左往右应该递增，即 $a[l_i]<a[l_{i+1}]$，且它们的并为 $[l,r]$。

若当前子树的左端点为 $l_p$，枚举右端点 $r_p$，需要满足 $a[r_p+1]>a[l_p]$ 或 $r_p=r$（否则下一棵子树的根节点将不满足题意）。

考虑它之前的若干兄弟子树（都是 $a[l]$ 的子节点，且根节点编号依次递增，且小于它）有多少种情况，不妨记 $g(l,i)$ 表示 $l\sim i-1$ 的树数量，且满足 $a[l]$ 的所有子节点标号最大的点小于 $a[i]$，那么此时 $[l_p,r_p]$ 就可以插到这棵树中。

转移方程为 $g(l,l_p)\times f(l_p,r_p)\rightarrow g(l,r_p+1)$，$f(l,r)=\sum\limits_{i=l}^{r}{g(l,i)\times f(i,r)}$。

我们需要枚举当前区间 $[l,r]$ 以及转移子区间 $[l_p,r_p]$，乍一看这是 $O(n^4)$ 的转移，但实际上不是。

当 $l$ 和 $l_p$ 固定时，对于不同的 $r$，$r_p\in [l_p,r]$，因此随着 $r$ 递增，对应 $r_p$ 的区间的前缀是相同的，只会在末尾增加新的满足条件的 $r_p$。因此只需要计算新的 $r_p$ 的贡献， $r_p$ 这一维就被均摊掉了。只需要时刻记录每个 $l,l_p$ 对应的 $r_p$ 上一次被更新到了什么位置，下一次继续向右拓展即可。

复杂度 $O(n^3)$。

**代码1：** https://atcoder.jp/contests/abc252/submissions/35802015

**题解2：** 左儿子右兄弟表示法

任意一棵树都可以根据左儿子右兄弟法转换成二叉树，这是一一对应的关系。

新树的限制为，任意点的右儿子（如果存在的话），编号需大于它本身。

不难发现，新树的任意节点 $x$ 的子树的DFS序都对应着原树上节点 $x$ 的子树的DFS序，这依然是个一一对应的关系。

用 $f(l,r)$ 表示DFS序上区间 $[l,r]$ 对应的子树数量。

那么 $f(l,r)=\sum\limits_{k\in [i+1,j+1]\land(a[k]>a[i]\lor k>j)}{f(l+1,k-1)\times f(k,r)}$，注意左右子树可以为空。

答案为 $f(2,n)$，因为 $a[2]$ 一定是 $a[1]$ 的左儿子，且 $a[1]$ 没有右儿子。

复杂度 $O(n^3)$。

**代码2：** https://atcoder.jp/contests/abc252/submissions/35802959

## ABC 257 G - Prefix Concatenation

**题意：** 给定字符串 $s,t$，求 $t$ 至少通过多少个 $s$ 的前缀拼凑而成。

**限制：** $\vert s\vert ,\vert t\vert \leq 5\times 10^5$

**链接：** https://atcoder.jp/contests/abc257/tasks/abc257_g

**题解：** 字符串哈希、线段树

记 $f(i)$ 为 $t$ 的前 $i$ 位前缀至少需要多少个 $s$ 的前缀，答案为 $f(\vert t\vert )$。

从左往右枚举起始位置 $i$，二分+哈希可以得到能与 $s$ 前缀匹配的右端点最大值 $r$（即$t[i:r]=s[1:r-i+1]$），则 $f(i-1)+1\rightarrow f(i)\sim f(r)$，这可以用线段树维护区间取 $\min$。

复杂度 $O(\vert t\vert \log{\vert t\vert })$。

**代码：** https://atcoder.jp/contests/abc257/submissions/35804755

## ABC 273 G - Row Column Sums 2

**题意：** 在 $n\times n$ 的方阵中填非负整数，要求第 $i$ 行的和为 $r_i$，第 $i$ 列的和为 $c_i$。求方案数。

**限制：** $n\leq 5000,0\leq c_i,r_i\leq 2$

**链接：** https://atcoder.jp/contests/abc273/tasks/abc273_g

**题解：** DP

显然满足 $\sum{c_i}=\sum{r_i}$ 时才有解，且每个格子只能填入 $0,1,2$。

定义 $f(i,c)$ 表示填完前 $i$ 行后，存在 $c$ 列的当前元素之和 $r_i'$ 满足 $r_i-r_i'=2$。

记 $\text{sum}=\sum\limits_{j=1}^{i}{r_j},\text{total}=\sum{r_i}$，则满足 $r_i-r_i'=1$ 的列数为 $\text{total}-\text{sum}-2c$。

不妨直接用 $(a,b,c)$ 表示满足 $r_i-r_i'=0,1,2$ 的列数。

现在考虑填入第 $i$ 行，分情况讨论：

- $r_i=0$：只能全填 $0$，$f(i,c)= f(i,c)$
- $r_i=1$：$2\rightarrow 1$ 或 $1\rightarrow 0$，即 $f(i,c)=f(i-1,c)\times (b+1)+f(i,c)\times (c+1)$
- $r_i=2$：$2\rightarrow 0$ 或 $2\rightarrow 1,2\rightarrow 1$ 或 $2\rightarrow 1,1\rightarrow 0$ 或 $1\rightarrow 0,1\rightarrow 0$：$f(i,c)=f(i-1,c)\times (c+1)+f(i-1,c)\times \binom{b+2}{2}+f(i-1,c+2)\times \binom{c+2}{2}+f(i-1,c+1)\times (c+1)$

复杂度 $O(n^2)$。

**代码：** https://atcoder.jp/contests/abc273/submissions/35805977

## ABC 254 G - Elevators

**题意：** 有 $n$ 栋楼，有 $m$ 部电梯，第 $i$ 部电梯 $(a_i,b_i,c_i)$ 表示在第 $a_i$ 栋楼有一部电梯可以在楼层 $[b_i,c_i]$ 间工作，从 $s$ 层移动到 $t$ 层花费的时间为 $\vert s-t\vert (b_i\leq s,t\leq c_i)$。如果当前位置没有电梯，则不能上下移动。此外在可以花费 $1$ 的时间移动到其他楼的同一层。现在有 $q$ 个询问，询问从第 $a$ 栋楼的第 $x$ 层出发，到第 $b$ 栋楼的第 $y$ 层的最短时间。

**限制：** $1\leq n,m,q\leq 2\times 10^5,1\leq b_i\leq c_i\leq 10^9$

**链接：** https://atcoder.jp/contests/abc254/tasks/abc254_g

**题解：** 倍增+DP

同一栋楼中，区间存在交集的若干部电梯可以合并成一部电梯。

移动是双向可逆的，因此不妨设 $x\leq y$，否则可以交换起点终点。

先假设有解，则答案由两部分组成，横向移动和竖向移动。

竖向移动所花费的 $y-x$ 的时间是必须的，且不再需要花费额外时间，因为到达 $y$ 层后可以直接横向移动到第 $b$ 栋楼。

现在只需要考虑需要横向移动多少次。

先让 $x$ 通过电梯到达第 $a$ 栋楼可以到达的最高层，不妨设为 $x'$，并让 $y$ 通过电梯到达第 $b$ 栋楼可以到达的最低层，不放设为 $y'$。

如果 $y'\leq x’$，则答案为 $y-x+(a\not=b)$。

否则先花时间 $1$，让 $x'$ 移动到其他楼中所有满足 $x'\in [b_i,c_i]$ 中的 $\max{c_i}$，并假设现在第 $p$ 部电梯上。

此时若满足 $y'\leq x'$，则再花费 $1$ 时间移动到第 $b$ 栋楼即可（显然此时不可能已经在第 $b$ 栋楼，不然在第一步移动后，应该已经满足 $y'\leq x'$ 并输出答案） 。

设 $f(i,j)$ 表示第 $i$ 部电梯横向移动 $j$ 次后可以到达的最高楼层对应的电梯编号。

那么此时还需要横向移动的次数为 $\min\limits_{c_{f(p,j)}\geq y'}{j}$。

显然 $f(i,*)$ 是单调递增的，因此可以倍增维护，将 $f(i,j)$ 修改为第 $i$ 部电梯横向移动 $2^j$ 次后可以到达的最高楼层对应的电梯编号。

那么先移动到小于 $y'$ 的最高楼层，再向上额外移动一次，并最终横向移动到第 $b$ 栋楼。

如果在此过程中无法移动到 $\geq y'$ 层，则无解。 

复杂度 $O((m+q)\log{m})$。

**代码：** https://atcoder.jp/contests/abc254/submissions/35816925

## ABC 222 G - 222

**题意：** 对于无限长的序列 $2,22,222,\cdots$，求何时第一次出现 $k$ 的倍数，或者判断无解 。一共  $t$ 组数据。

**限制：** $t\leq 200,t\leq 10^8$

**链接：** https://atcoder.jp/contests/abc222/tasks/abc222_g

**题解：** 整除

$\frac{a}{b}x\equiv0\pmod{p}$ 等价于 $ax\equiv0\pmod{bp}$，而 $ax\equiv0\pmod{p}$ 等价于 $x\equiv{0}\pmod{\frac{p}{\gcd(p,a)}}$。

序列的第 $i$ 项可以表示为 $\frac{2}{9}(10^i-1)$，若 $\frac{2}{9}(10^i-1)\equiv 0\pmod k$，则 $10^i\equiv 1\pmod{k'}$，若 $k$ 为奇数，则 $k'=9k$，否则 $k'=\frac{9}{2}k$。

根据欧拉定理，$10^{\varphi(k)}\equiv1\pmod{k}$，因此若有解， $i$ 一定是 $\varphi(k')$ 的因数，逐个验证即可。

复杂度 $O(t\sqrt{k}\log{k})$。

**代码：** https://atcoder.jp/contests/abc222/submissions/35841443

## ABC 252 G - Swap Many Times

**题意：** 给定 $n$，将 $(a,b)(1\leq a<b\leq n)$ 按照字典序排列。给定 $l,r$，假设其对应的子区间为 $(a_l,b_l),\cdots,(a_r,b_r)$，则将初始状态为 $v=(1,2,\cdots,n)$ 的排列逐个进行 $\text{swap}(v[a_{l}],v[a_{r}])$ 。求 $v$ 经过 $r-l+1$ 次操作后的状态。

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc252/tasks/abc253_g

**题解：** 模拟

$(1,\*),(2,\*),\cdots,(n-1,*)$ 分别有 $n-1,n-2,\cdots,1$ 个，因此很容易得到 $(a_l,b_l)$ 和 $(a_r,b_r)$。

那么操作可以分成 $(a_l,b_l),\cdots,(a_l,n-1)$，$(a_{l}+1,\*),\cdots,(a_{r}-1,\*)$，$(a_r,1),\cdots,(a_r,b_r)$ 这三部分。

首位两部分只有 $O(n)$ 项，可以直接模拟。现在考虑中间的部分。

经过观察不难发现，完整经过 $(x,\*)$ 这一轮操作，对原排列的影响为将 $v[x],\cdots,v[n]$ 向后轮换旋转了一位（视为一个环），元素的相对顺序关系没有发生变化，而 $(x+1,*)$ 则只是原环删去 $v[x]$ 。因此这部分操作可以直接用一个双端队列维护首尾元素。

注意特殊处理一下 $a_l=a_r$ 的情况，此时只有一段 $(a_l,b_l),\cdots,(a_l,b_r)$。

复杂度 $O(n)$。

**代码：** https://atcoder.jp/contests/abc253/submissions/35839378

## ABC 274 G - Security Camera 3

**题意：** 给定一个 $h\times w$ 的网格图，每个点要么是空地要么是障碍。现在可以在空地上放置若干个激光器，激光器会向指定方向（上下左右）发射激光，点亮所有途经空地，直到遇到障碍物或者离开边界为止。每个空地可以放多个激光器，激光途经的激光器不会影响它的传播。求最少放置多少个激光器，使得所有空地都被点亮。

**限制：** $1\leq h,w\leq 300$

**链接：** https://atcoder.jp/contests/abc274/tasks/abc274_g

**题解：** 最大流

虽然激光器可以有四个方向，但显然上/下等价，左/右等价，我们不妨假定所有激光器都是向下或者向右发射的。

本着贪心的思想，放置向下激光器的位置要么是上边界处，要么正上方的格子恰好是障碍，否则向上移动一格显然更优。向右的激光器同理，应该放到尽可能靠左的位置。

因此每个空地 $(i,j)$ 都可以计算出【若被向右的激光器点亮，激光器的位置】$f(i,j)$，和【若被向下的激光器点亮，激光器的位置】$g(i,j)$。

然后就是经典的最小割建图模型：

- $s\to f(i,j)$，流量为 $1$
- $f(i,j)\to g(i,j)$，流量为 $\infty$
- $g(i,j)\to t$，流量为 $1$

正确性：对于任意点 $(i,j)$，两个激光器 $f(i,j)$ 和 $g(i,j)$ 必须至少选择一个。考虑图上路径 $s\to f(i,j)\to g(i,j)\to t$，由于 $f(i,j)\to g(i,j)$ 是 $\infty$，因此 $s\to f(i,j)$ 和 $g(i,j)\to t$ 至少被割去其一，也就满足了条件。

答案为 $\text{maxflow}(s,t)$，由于边权为 $1$，dinic的复杂度为 $O(hw)^{1.5}$。

**代码：** https://atcoder.jp/contests/abc274/submissions/35935072

## ABC 250 G - Stonks

**题意：** 初始状态有无限多的钱炒股，已知接下来 $n$ 天的股价。每天可以买/卖一股（如果有的话）。求 $n$ 天后的最大收益。  

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc250/tasks/abc250_g

**题解1：** 堆

所有交易都是一买一卖的匹配，$(p_i,p_j)(p_j>p_i,i<j)$，那么总收益为 $\sum{p_j-p_i}$。

考虑匹配之间的传递关系，若已有匹配 $(p_a,p_b)$，收益为 $p_b-p_a$。由于 $p_c>p_b$，将其替换成 $(p_a,p_c)$ 现在更优，此时额外的收益 $(p_c-p_a)-(p_b-p_a)=p_c-p_b$，此时 $p_b$ 又待匹配。$p_b$ 存在一个被减数和减数之间的转换。

因此用一个小根堆维护所有待匹配的数和它是否匹配的pair，从左往右扫描，记当前值为 $x$。取出堆顶 $<y,f>$。

- 若 $y<x$，则贡献增加 $x-y$
  - 若 $f=1$，则将 $<y,0>$ 插入回堆中（类似 $p_b$ 与 $p_c$ 的状态）
  - 将 $<x,1>$ 插入堆中
- 否则将 $<x,0>$ 插入堆中

复杂度 $O(n\log{n})$。

upd：看了眼官方题解，它是直接如果 $x>y$ ，就插入 $2$ 个 $x$ ，否则插入 $1$ 个。本质上是一样的，只是不用维护pair了。

**题解2：** 线段树

令 $a_i=1,0,-1$ 表示第 $i$ 天买/不操作/卖股票，$s_i$ 为 $a_i$ 的前缀和，表示第 $i$ 天的股票数量，显然要满足 $\forall i,s_i\geq 0$。

先让 $\forall a_i=1$，然后再修改，使得 $a_i=0$ 或 $-1$，显然优先选择价格高的修改。

按照{股票价格，天数}的pair降序排列，由于修改后时刻要保证 $\forall s_i\geq 0$，因此要满足右区间 $s$ 的最小值 $\geq 2$ 或者 $\geq 1$。修改后会让右区间整体 $-2$ 或者 $-1$，这可以用线段树区间加法/区间最小值来维护。

最后答案为 $\sum{a_i\times p_i}$。

复杂度 $O(n\log{n})$。

**代码1：** https://atcoder.jp/contests/abc250/submissions/35943574

## ABC 251 G - Intersection of Polygons

**题意：** 给定 $n$ 个点的凸多边形 $P_0=\{(x_i,y_i)\}$ 和 $m$ 个偏移量 $(a_i,b_i)$。$P_i$ 表示将 $P_0$ 在 $x$ 方向移动 $a_i$，在 $y$ 方向移动 $b_i$ 后的新多边形。给定 $q$ 个坐标 $(p_i,q_i)$，判断每个点是否同时在多边形 $P_1,\cdots,P_m$ 中。

**限制：** $n\leq 50,m,q\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc251/tasks/abc251_g

**题解：** 计算几何

判断点 $q=(a,b)$ 是否在凸多边形 $\{p_i=(x_i,y_i)\}$ 中的方法是逐个判断 $\vec{va_i}=(q-p_i),\vec{vb_i}=(p_{i+1}-p_i)$ 的叉积是否全部非负。

不难发现对于任意 $P_i$，其 $\vec{vb_i}$ 都是相同的。

对于 $P_j=\{p_{i,j}=(x_i+a_j,y_i+b_j)\}$，即枚举 $i$，判断 $(q-p_{i,j}) \times \vec{vb_i}\geq 0$，等价于 $\vec{q}\times \vec{vb_i}\geq \vec{p_{i,j}}\times \vec{vb_i}$。

可以 $O(nm)$ 预处理出所有 $\vec{p_{i,j}}\times \vec{vb_i}$，然后对于每个 $i$，计算 $r_i=\max\limits_{j=1}^{m}{\vec{p_{i,j}}\times \vec{vb_i}}$。

查询时，只需要枚举 $i$，判断 $\vec{q}\times \vec{vb_i}\geq r_i$ 即可，复杂度 $O((m+q)\times n)$。

**代码：** https://atcoder.jp/contests/abc251/submissions/35953076

## ABC 217 G - Groups

**题意：** 将 $1\sim n$ 分成 $k$ 组，要求每组非空，且同余 $m$ 的数不能属于一组，对于 $k=1\sim n$，求方案数。

**限制：** $m\leq n\leq 5000$

**链接：** https://atcoder.jp/contests/abc217/tasks/abc217_g

**题解：** 计数

先将 $1\sim n$ 按照 $\bmod m$ 分组，显然有 $m-n\bmod m$ 个集合大小为 $\lfloor\frac{n}{m}\rfloor$，$n\bmod m$ 个集合大小为 $\lceil\frac{n}{m}\rceil$。

$f(i)$ 表示分成 $i$ 组的方案数，则：

$$
f(i)=\frac{(A_k^{\lfloor\frac{n}{m}\rfloor})^{m-n\bmod m}\times (A_k^{\lceil\frac{n}{m}\rceil})^{n\bmod m}-\sum\limits_{j=0}^{i-1}f(j)\frac{i!}{(i-j)!}}{i!}
$$

复杂度 $O(n^2)$。

**代码：** https://atcoder.jp/contests/abc217/submissions/35956221

## ABC 217 F - Cleaning Robot

**题意：** 从 $(0,0)$ 出发，给定字符串 $S$，`LRUD` 表示移动方向，每次移动一格。将 $S$ 重复 $k$ 次，求过程中有多少个点被访问。

**限制：** $\vert S\vert \leq 2\times 10^5,k\leq 10^{12}$

**链接：** https://atcoder.jp/contests/abc219/tasks/abc219_f

**题解：** 同余

维护前 $\vert S\vert $ 步对应的点集 $\{(x_i,y_i)\}$，也包括起点 $(0,0)$。

记 $(a,b)$ 为 $\vert S\vert $ 步后的坐标，则通过 $V_1$，可以得到 $V_2=\{(x_i+a,y_i+b)\},\cdots,V_k=\{(x_i+(k-1)a,y_i+(k-1)b\}$。

那么答案就是这 $k$ 个集合并集的大小。

显然，如果 $a=0,b=0$，则答案为 $\vert V_1\vert $。

若 $a=0$，交换 $x,y$ 坐标，即可保证 $a\not=0$。

考虑哪些点会重复，假设 $(x_i+pa,x_j+pb)=(x_j+qa,y_j+qb)$，则$x_i-x_j\equiv 0\pmod{a}$。因此可以按照 $x_i\bmod a$ 分组，同一组中的点才可能重复。

枚举 $p\in[0,a)$，将所有 $x_i\bmod a=p$ 坐标写成 $(x_i=c_i\times a+p,y_i=c_i\times b+q)$ 的形式，不难发现，$(p,q)$ 相同的坐标构成一个无限长的序列，$(x_i,y_i)$ 在其中的下标为 $c_i$，每经过一轮，就会移动到序列的下一个位置。

因此对于每个 $(p,q)$，将 $c_i$ 升序排序，则每个坐标不与下一个坐标重复的轮数为 $\min(c_{i+1}-c_i,k)$，$c_i$ 最大的坐标将独享 $k$ 轮。

因此答案为 $\sum\limits_{(p,q)}\sum\limits_{i}\min(c_{i+1}-c_i,k)$。

复杂度 $O(\vert S\vert \log{S})$。

**代码：** https://atcoder.jp/contests/abc219/submissions/35978816

## ABC 218 G - Game on Tree 2

**题意：** 给定一棵树，每个点有权值。从 $1$ 号点出发，Alice和Bob轮流移动到一个未访问的点，无法移动时游戏结束。游戏分数为经过所有点的权值的中位数。Alice想要最大化，Bob想要最小化，求最终答案。

**限制：** $n\leq 10^5$

**链接：** https://atcoder.jp/contests/abc218/tasks/abc218_g

**题解：** 博弈，中位数

每局游戏都是从 $1$ 开始，在某个叶子节点结束，对应的中位数是确定的。然后从下往上做经典的min/max树形DP即可。

至于如何求中位数，可以采用平衡树或者双multiset。

复杂度 $O(n\log{n})$。

**代码：** https://atcoder.jp/contests/abc218/submissions/35981987

## ABC 212 G - Power Pair

**题意：** 给定质数 $p$，求满足 $\exists n,x^n\equiv y\pmod{p}$数对 $(x,y)(0\leq x,y< p)$ 的数量。 

**限制：** $p\leq 10^{12}$

**链接：** https://atcoder.jp/contests/abc212/tasks/abc212_g

**题解：** 原根

由于 $p$ 是质数，因此必然存在原根 $r$，满足 $r^1,r^1,\cdots,r^{p-1}$ 遍历 $[1,p-1]$ 中的所有值。

因此 $x$ 和 $y$ 都可以写成 $r^a,r^b$ 的形式。

问题转换成 $\exists n, r^{an}\equiv r^b\pmod{p}$，等价于 $\exists n,an\equiv b\pmod{p-1}$。

枚举 $a$，则 $b$ 的方案数为 $\frac{p-1}{\gcd(p-1,a)}$，因此答案为 $\sum\limits_{a=1}^{p-1}{\frac{p-1}{\gcd(p-1,a)}}$。

显然 $\gcd(p-1,a)$ 的取值只能是 $p-1$ 的因数，因此枚举 $p-1$ 的因数 $k$，问题转换成求 $\gcd(p-1,a)=k$ 的 $a$ 的数量。

写成乘积的形式， $p-1=k\times g_k,a=k\times a_k$，显然 $\gcd(g_k,a_k)=1$，因此 $a$ 的数量为 $\varphi(\frac{p-1}{k})$。

因此答案为 $\sum\limits_{k\vert p-1}{\frac{p-1}{k}\times \varphi(\frac{p-1}{k})}$。

**代码：** https://atcoder.jp/contests/abc212/submissions/35990374

## ABC 215 G - Colorful Candies 2

**题意：** 有 $n$ 个球，第 $i$ 个球有颜色 $c_i$。现在其中随机选择 $k$ 个球，在 $\binom{n}{k}$ 种方案中随机选择，对于 $k=1,2,\cdots,n$，分别求其中不同颜色球个数的期望数。

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc215/tasks/abc215_g

**题解：** 期望

先统计每种颜色的球个数，不妨设有 $m$ 种颜色，第 $i$ 种颜色的球有 $n_i$ 个。

设随机变量 $X_i$ 表示第 $i$ 种颜色的球是否被选择，则期望为 $E[X_i]$，不同颜色的球个数为 $E[\sum\limits_{i=1}^{m}{X_i}]$，根据期望的线性性，$E[\sum{X_i}]=\sum{E[X_i]}$。

$E[X_i]=\frac{\binom{n}{k}-\binom{n-n_i}{k}}{\binom{n}{k}}$，因此答案为 $\sum\limits_{i=1}^{m}{\frac{\binom{n}{k}-\binom{n-n_i}{k}}{\binom{n}{k}}}$，但这样算是 $O(m)$ 的。

由于一共 $n$ 个球，因此 $\sum{n_i}=n$，则 $n_i$ 的不同取值至多 $O(\sqrt{n})$ 种。而 $n_i$ 相同的颜色，$E[X_i]$ 相同，可以合并计算。

总复杂度 $O(n\sqrt{n})$。

**代码：** https://atcoder.jp/contests/abc215/submissions/35990728

## ABC 216 G - 01Sequence

**题意：** 给定 $01$ 序列的长度 $n$ ，以及 $m$ 个限制条件 $(l_i,r_i,x_i)$，要求在序列区间 $[l_i,r_i]$ 中至少有 $x$ 个 $1$。构造出 $1$ 数量最少的序列。

**限制：** $n\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc216/tasks/abc216_g

**题解1：** 贪心

显然 $1$ 要尽可能的往后放，这样才能一次性满足多个限制条件。

将 $(l_i,r_i,x_i)$ 按照 $r_i$ 升序排序。假设当前区间还需要 $y_i$ 个 $1$ 才能满足限制，则从 $r_i$ 开始向左，将最靠右的 $y_i$ 个未更新位置置成 $1$。

$y_i$ 通过树状数组维护，未更新位置通过 set 维护。

复杂度 $O(n\log{n})$。

**题解2：** 差分约束

最少 $1$ 等价于最多 $0$。

设 $b_i$ 为 $a_1,\cdots,a_i$ 中 $0$ 的个数，$b$ 显然满足以下限制：

- $b_0=0$

- $b_i\leq b_{i+1}$
- $b_{r_i}-b_{l_i-1}\leq r_i-l_i+1-x_i$
- $b_i+1\geq b_{i+1}$

转换成差分约束系统，形如 $b_i-b_j\leq k$ 的限制就连接一条单向边 $u\to v$，边权为 $k$。

跑最短路，则最终的序列 $a_i=1-(b_{i+1}-b_i)$。

复杂度 $O(n\log{n})$。

**代码1：** https://atcoder.jp/contests/abc216/submissions/35993073

**代码2：** https://atcoder.jp/contests/abc216/submissions/35992897

## ABC 260 G - Scalene Triangle Area

**题意：** 给定 $n\times n$ 的网格图 和参数 $m$，若 $a[i][j]$ 为 $1$，则它会覆盖所有满足 $i\leq s\leq n,j\leq t\leq n,s-i+\frac{t-j}{2}<m$ 的格子 $(s,t)$。 $q$ 个询问，每次查询格子 $(x_i,y_i)$ 被覆盖了几次。

**限制：** $n\leq 2000$

**链接：** https://atcoder.jp/contests/abc260/tasks/abc260_g

**题解：** 差分

考虑 $(i,j)$ 会影响哪些格子，不考虑溢出的情况下，为 $i:[j,j+2\times m-1],i+1:[j,j+2\times m-3],\cdots,i+m-1:[j,j+1]$。

不难发现右端点是一个差为 $2$ 的等差数列。因此考虑打标记，最后做前缀和。

左端点则是固定的，连续 $m$ 行。

打两类标记，一类是右端点标记 $r$，一类是左端点标记 $l$。

对于右端点，$(i,j)\to (i+1,j-2)$，因此在 $(i,j+2\times m-1)$ 处 $+1$，在 $(i+m,j-1)$ 处 $-1$。

对于左端点，$(i,j)\to (i+1,j)$，因此在 $(i,j-1)$ 处 $-1$，在 $(i+m,j-1)$ 处 $-1$。

然后分别对 $l,r$ 根据转移方向做前缀和。

最后每行从右往左再对 $l,r$ 做后缀和，格子 $(x,y)$ 被覆盖的次数为 $l(x,y)+r(x,y)$。

复杂度 $O(n^2)$。

**代码：** https://atcoder.jp/contests/abc260/submissions/35995973

## ABC 220 G - Isosceles Trapezium

**题意：** 给定平面上 $n$ 个点的坐标及其权值。选择其中四个点形成等腰梯形，求权值之和的最大值。

**限制：** $n\leq 1000$

**链接：** https://atcoder.jp/contests/abc220/tasks/abc220_g

**题解：** 计算几何

等腰梯形的平行对边满足以下条件：

- 垂直平分线相同
- 中点坐标不同（否则两条边可能在同一条直线上）

因此枚举所有直线，求出对应的中点坐标和垂直平分线即可，复杂度 $O(n^2)$。

**代码：** https://atcoder.jp/contests/abc220/submissions/36012390

## ABC 259 G - Grid Card Game

**题意：** 给定 $h\times w$ 的网格图，每个格子上有权值 $a_{i,j}$。选择若干行若干列，并最大化被选格子的权值之和。要求若 $a_{i,j}<0$，第 $i$ 行和第 $j$ 列不能同时选。

**限制：** $h,w\leq 100$

**链接：** https://atcoder.jp/contests/abc259/tasks/abc259_g

**题解：** 最小割

由于权值有正有负，考虑先选择所有正权值的格子，然后通过最小的代价将其调整为合法方案。

设 $f(i)$ 为 $\sum\limits_{j=1}^{w}{a_{i,j}}$，表示如果选择第 $i$ 行的收益，$h_i$ 为 $\sum\limits_{j=1}^{h}{a_{j,i}}$，表示选择第 $j$ 列的收益。

显然如果收益为负数，则不会选，记 $\sum^+=\sum{\max(f(i),0)}+\sum{\max(g(i),0)}$。

如下方式建图：

- 如果 $f(i)\geq 0$，$s\to i$ ，流量为 $f(i)$
- 如果 $a_{i,j}\geq 0$，$i\to j'$，流量为 $a_{i,j}$，否则 $i\to j'$，流量为 $\infty$
- 如果 $g(j)\geq 0$，$j'\to t$，流量为 $g(j)$

答案为 $\sum^+-\text{maxflow}(s,t)$。

复杂度 $O(n^4)$。

**代码：** https://atcoder.jp/contests/abc259/submissions/36014518

## ABC 213 G - Connectivity 2

**题意：** 给定一个 $n$ 个点 $m$ 条边的无向图。对于 $k=2,\cdots,n$，求有多少个子图满足 $1$ 和 $k$ 属于同一个连通块。

**限制：** $n\leq 18$

**链接：** https://atcoder.jp/contests/abc213/tasks/abc213_g

**题解：** 容斥、状压

记 $f(S)$ 表示点集 $S$ 构成一个连通块的子图数量，$g(S)$ 为点集 $S$ 的子图数量。

显然 $g(S)$ 为 $2^{E(S)}$，$E(S)$ 为两端点都属于 $S$ 集合的边数量。

采用枚举子集和容斥的方式计算 $f(S)$，枚举其中的点 $v$，$f(S)=g(S)-\sum\limits_{v\in T\subsetneqq S}f(T)\times g(S\setminus T)$。

则答案为 $\sum\limits_{k\in S}{f(S)\times g(S')}$。

复杂度 $O(n^22^n+3^n)$。

**代码：** https://atcoder.jp/contests/abc213/submissions/36015734

## ABC 255 G - Constrained Nim

**题意：** $n$ 堆石子，第 $i$ 堆石子有 $a_i$ 个。两人轮流取石子，规则为Nim游戏，并额外增加了 $m$ 个限制 $(x_i,y_i)$，即当某堆石子个数恰好为 $y_i$ 时，不能从中取走 $x_i$ 个。求最后的胜者。

**限制：** $n,m\leq 2\times 10^5,x_i\leq y_i\leq 10^{18}$

**链接：** https://atcoder.jp/contests/abc255/tasks/abc255_g

**题解：** SG函数

最后的结果为每堆石子个数的SG函数的异或和。

记 $p(n)=\{n-m\vert (n,m)\in (x_i,y_i)\}$ ，即 $n$ 不能从哪些点转移。

设 $f(n)$ 为石子个数恰好为 $n$ 时对应的SG函数， $g(n)=\max\limits_{i=1}^{n}f(n)$。

注意到一个重要性质，$0,1,\cdots,g(n)$ 在 $f(0),f(1),\cdots,f(n)$ 中都至少出现过一次（不然根据mex的定义无法递增）。

根据SG函数定义，$f(n)=\text{mex}\{f(v_n)\}+1$，其中 $v_n$ 为 $n$ 能转移到的数。

设 $S=\{y_1,\cdots,y_m\}$，当 $n\notin S$ 时，由于 $v_n=\{1,2,\cdots,n\}$，得 $f(n)=\text{mex}\{f(0),\cdots,f(n-1)\}=g(n-1)+1$。

统计 $0,1,2,\cdots, g(n)$ 的出现次数，则当求 $n\in S$ 的 $f(n)$ 时，只需要知道去掉 $\{f(v)\vert v\in{p_n}\}$ 后，未出现的最小正整数是谁。

由于值域是LLONG，我们不可能维护所有数出现的次数，但是根据前文的性质，我们只需要维护每个值除 $1$ 以外的额外出现次数。而这些异常点的数量之多只有 $m$ 个。

复杂度 $O(m\log{m})$。

**代码：** https://atcoder.jp/contests/abc255/submissions/36018089

## ABC 212 F - Greedy Takahashi

**题意：** $n$ 个城市之间有 $m$ 辆公交车。第 $i$ 辆公交车从 $s_i$ 城市出发，到 $t_i$ 城市，出发时间为 $a_i+0.5$，到达时间为 $b_i+0.5$。旅客在某时刻到达某城市后，将乘坐最早的公交离开，如果之后没有公交将停留在该城市，数据保证 $s_i$ 互不相同。现在有 $q$ 个询问，查询旅客从 $u_i$ 城市出发，旅行时间为 $[x_i,y_i]$，求在 $y_i$ 时刻的位置。

**限制：** $n,m,q\leq 2\times 10^5$

**链接：** https://atcoder.jp/contests/abc212/tasks/abc212_f

**题解：** 倍增

维护 $f[i][j]$ 表示从第 $i$ 辆公交车出发，换乘 $2^j$ 辆公交车后的位置。

复杂度 $O((q+m)\log{m})$。

**代码：** https://atcoder.jp/contests/abc212/submissions/36027453
