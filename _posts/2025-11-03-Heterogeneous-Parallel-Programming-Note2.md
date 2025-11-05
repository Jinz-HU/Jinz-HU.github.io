---
layout: post
title: Heterogeneous Parallel Programming —— 笔记4
subtitle: Heterogeneous Parallel Programming
date: 2025-11-03
author: Jinz
header-img: img/post-bg5.jpg
catalog: true
tags:
  - Parallel Programming

---

# Reduction（规约）

**Reduction（规约）是指将一个大集合（如数组、列表）中的所有元素，通过一个二元操作符，合并成单个值的过程。**


## 总操作量与“工作效率”

输入N个操作数，**并行Reduction**的总操作次数为：

`(1/2)N + (1/4)N + (1/8)N + … (1) = (1- (1/N))N = N-1 operations`

## 算法步长（时间复杂度）

```text
         [1,2,3,4,5,6,7,8]  (初始数据)
          /     |     |     \     (划分)
        [1,2] [3,4] [5,6] [7,8] (每个线程局部计算)
          |     |     |     |
          3     7     11    15   (局部结果)
          \     /     \     /    (第一轮并行合并)
           10          26         (中间结果)
            \         /          (第二轮并行合并)
              36                 (最终结果)
```

*算法像一棵二叉树一样合并，从 N 个值合并到 1 个值，需要的高度（步数）是 log₂(N)*

##  并行度与资源效率

- 平均并行度：

  - 总操作量是 N-1。

  - 完成这些操作需要 log(N) 个时间步。

  - 所以，平均每个时间步执行的操作数（即平均并行度）为 (N-1) / log(N)。

  - 对于 N=1,000,000，平均并行度 ≈ 999,999 / 20 ≈ 50,000。这意味着平均下来，你需要 50,000 个处理器来保持忙碌。

- 峰值资源需求

  - 第一步需要对 N/2 个元素对同时进行操作。对于 N=1,000,000，这需要 **500,000** 个处理器

> *为峰值性能支付了巨额成本，但平均利用率却很低*

> *特性： 工作高效、时间高效、资源低效*

**实际的并行 Reduction 实现是分层的：**

首先，在有限的处理器上，每个处理器先串行处理一大块数据，计算出一个局部结果。

然后，再对这些少得多的局部结果应用树形 Reduction。

*例如，在GPU上对1,000,000个数求和，可能会先让每个GPU线程块处理16,384个数，产生约61个局部和，然后再对这61个局部和进行树形合并。*


## 减少规约中的--控制分歧

```cpp
// 版本1：高控制分歧（不推荐）
__global__ void reduce_high_divergence(int *input, int *output) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    
    // ... 数据加载到 sdata ...
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        // 高分歧点：线程束内的线程会进入不同的if分支
        if (tid % (2 * s) == 0) { 
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // ... 写入结果 ...
}
```
> 当 s = 1 时，只有 tid % 2 == 0 的线程（即tid为0,2,4,6,...）执行加法，其他线程闲置。
> 
> 在一个32线程的线程束中，16个线程执行if块，16个线程执行else（空操作），导致线程束分化为两条路径，必须串行执行。

```cpp
// 版本2：低控制分歧（推荐）
__global__ void reduce_low_divergence(int *input, int *output) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    
    // ... 数据加载到 sdata ...
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {  // 关键改变：只有前s个线程工作
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // ... 写入结果 ...
}
```

> 当 s = blockDim.x/2 时，前一半线程（tid < s）都执行加法，后一半线程（tid >= s）都不执行。
>
> 在线程束级别：由于线程ID是连续的，前几个线程束可能全部活跃或全部闲置，中间最多只有一个线程束会出现分化。
>
> 例如：块大小256，第一个循环 s=128：
> 
> 线程束0（tid 0-31）：全部满足 tid < 128，无分歧
> 线程束1（tid 32-63）：全部满足 tid < 128，无分歧
> 线程束2（tid 64-95）：全部满足 tid < 128，无分歧
> 线程束3（tid 96-127）：全部满足 tid < 128，无分歧
> 线程束4-7（tid 128-255）：全部不满足条件，无分歧
> **只有在s的值使得条件边界落在某个线程束内部时，才会出现一个线程束的分化**

# Scan(Prefix Sum)

**给定一个输入数组 `[a₀, a₁, a₂, ..., aₙ₋₁]`，前缀和产生一个输出数组，其中每个元素是输入数组中到该位置为止所有元素的和**

包含性前缀和：

```text
输入:  [3, 1, 7, 0, 4, 1, 6, 3]
输出:  [3, 4, 11, 11, 15, 16, 22, 25]
```

排他性前缀和：

```text
输入:  [3, 1, 7, 0, 4, 1, 6, 3]  
输出:  [0, 3, 4, 11, 11, 15, 16, 22]
```

## 串行Scan实现

```cpp
// 包含性前缀和 - 串行版本
void sequential_scan(int* input, int* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}
```
**时间复杂度：O(n)，顺序执行**

## 并行Scan实现

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103231418.png)

### Interleaved Reduction Trees（交错归约树）

**数据加载：全局内存 → 共享内存**

**多轮步长迭代（共$\log_2n$轮，n=8时为 3 轮）**

- 第 1 轮：步长 stride=1活跃线程处理索引 1~7，每个线程 j 将自身位置 j 与前 1 个位置 j-1 的元素相加，结果写回位置 j。

- 第 2 轮：步长 stride=2步长翻倍为 2，活跃线程处理索引 2~7，每个线程 j 将自身位置 j 与前 2 个位置 j-2 的元素相加。

- 第 3 轮：步长 stride=4步长再次翻倍为 4，活跃线程处理索引 4~7，每个线程 j 将自身位置 j 与前 4 个位置 j-4 的元素相加。

**结果写回：共享内存 → 全局内存**

### Blelloch Scan算法

**Reduce (上扫)**

```cpp
// 类似于归约，但保存所有中间结果
for (d = 0; d <= log₂(n)-1; d++) {
    for all k in parallel where k % 2^(d+1) == 2^(d+1)-1 {
        input[k] += input[k - 2^d]
    }
}

// 第一阶段：向上传播（Up-Sweep）
for (int s = 1; s < blockDim.x; s <<= 1) {
    int idx = (tid + 1) * 2 * s - 1;  // 计算当前步长下需要更新的索引
    if (idx < blockDim.x) {
        sdata[idx] += sdata[idx - s];
    }
    __syncthreads();
}
```

```text
初始:   3   1   7   0   4   1   6   3
d=0:    3   4   7   7   4   5   6   9  (相邻对相加)
d=1:    3   4   7   11  4   5   6   14 (间隔2相加)  
d=2:    3   4   7   11  4   5   6   25 (间隔4相加)
```

**DownSweep (下扫)**

```cpp
input[n-1] = 0  // 对于排他性扫描
for (d = log₂(n)-1; d >= 0; d--) {
    for all k in parallel where k % 2^(d+1) == 2^(d+1)-1 {
        int temp = input[k]
        input[k] = input[k + 2^d]
        input[k + 2^d] += temp
    }
}

// 第二阶段：向下传播（Down-Sweep）
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    int idx = (tid + 1) * 2 * s - 1;  // 计算当前步长下需要更新的索引
    if (idx < blockDim.x) {
        int temp = sdata[idx - s];
        sdata[idx - s] = sdata[idx];
        sdata[idx] += temp;
    }
    __syncthreads();
}
```

```text
初始:   3   4   7   11  4   5   6   0  (最后一个置0)
d=2:    3   4   7   0   4   5   6   11  (传播间隔4)
d=1:    3   0   7   4   4   11  6   16  (传播间隔2)
d=0:    0   3   4   11  11  15  16  22 (传播间隔1)
```


| 概念          | 本质属性                   | 核心作用                                                                                 | 覆盖范围                         |
| ------------- | -------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------- |
| Blelloch 算法 | 并行前缀和的**完整框架**   | 定义“向上归约（Up-Sweep）+ 向下传播（Down-Sweep）”两阶段流程，解决并行前缀和的逻辑正确性 | 从“数据加载”到“结果输出”的全流程 |
| 交错归约树    | 并行归约的**核心实现模块** | 是 Blelloch 算法“向上归约阶段”的高效实现方式，解决“如何并行合并局部数据”的问题           | 仅覆盖“向上归约”这一关键阶段     |



---
> **大规模输入向量的并行扫描问题: 分层扫描**