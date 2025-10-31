---
layout: post
title: Heterogeneous Parallel Programming —— 笔记2
subtitle: Heterogeneous Parallel Programming
date: 2025-10-31
author: Jinz
header-img: img/wallhaven-jxp3mw.jpg
catalog: true
tags:
  - Parallel Programming

---

# 线程调度

## GPU 中线程块（Thread Blocks）执行机制

线程以块（block）为单位被分配到 **流多处理器（SM）**，线程块是最小的调度单位

**每个 SM 最多可承载8 个线程块**，SM维护线程 / 块的索引，负责线程执行的调度与管理

> SM 内部包含**流处理器（SP）**和**共享内存（Shared Memory）**，这些组件共同支撑线程块的并行计算

## GPU 流多处理器（SM）中 warp（线程束）调度机制

**流多处理器（SM）采用 “零开销 warp 调度” 机制**

 warp 的下一条指令的操作数已准备好可供使用时，该 warp 就具备了执行资格，具备执行资格的 warp 会按照优先级调度策略被选中，其内部的所有线程会执行同一条指令

## GPU 中线程块如何划分为线程束

- 线程块会被划分为多个 warps
- warps 尺寸（即一个 warp 包含多少线程）会随 GPU 架构变化（**通常包含 32 个线程**）
- 线程之间存在依赖关系，必须使用__syncthreads()来保证同步，不要依赖 warps 内部或 warps 之间的执行顺序

# Control Divergence（控制分歧）

*同一线程束（warp）内的线程因执行不同分支（如if-else、循环条件等）而导致的并行效率下降问题*

当同一 warp 内的线程遇到分支指令（如if (condition)）时，若部分线程满足条件、部分不满足，就会出现控制分歧—— 满足条件的线程执行一个分支，不满足的线程执行另一个分支

> 假设一个 warp 有 32 个线程，其中 16 个满足if条件，16 个不满足。GPU 会先让 16 个线程执行if分支，剩下的 16 个线程执行else分支（或跳过）。这一过程中，warp 的并行度从 32 降到了 16（甚至更低），执行时间翻倍

```
Analysis for vector size of 1,000 elements
All threads in Blocks 0, 1, and 2 are within valid range
  i values from 0 to 767
Block 3 will have control divergence
  1st group i values from 768-999
  2nd group i values from 1000-1023
Performance penalty to Blocks with no divergence is very small
Only the last Block will have divergence
Overall performance penalty is small as long as there are large number of elements.
```

以16*16的线程块为例，0、1、2没有控制分歧；

Block 3出现控制分歧： 768-999 和 1000-1023分别处理。

> 当处理的数据集尺寸不是线程块大小的整数倍时，仅最后一个线程块会因 “有效 / 无效元素分支” 出现控制分歧。 **只要数据量足够大，这种分歧对整体性能的影响可忽略**

# 内存模型（Memory Model）

| 内存类型 | 访问速度             | 作用域                | 容量                   | 关键特性与使用场景                                                                                 |
| -------- | -------------------- | --------------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| 寄存器   | 极快（纳秒级）       | 线程私有              | 每个线程有限           | 存储线程私有变量，如循环索引、临时计算结果。需避免“寄存器溢出”（否则会降级为局部内存，性能骤降）。 |
| 共享内存 | 很快（接近寄存器）   | 线程块（Block）私有   | 每个 Block 约几十 KB   | 同一 Block 内的线程共享数据，适合临时数据复用（如矩阵分块计算）。需避免“银行冲突”以最大化带宽。    |
| 局部内存 | 慢（与全局内存相当） | 线程私有              | 大（全局内存的一部分） | 存储寄存器溢出的变量、大数组或动态索引数组。本质是全局内存的一部分，访问延迟高，需尽量避免。       |
| 全局内存 | 慢（百纳秒级）       | 所有线程（Grid 全局） | 大（GB 级）            | 存储跨 Block 共享的大规模数据。需通过内存合并（Coalescing）和数据对齐优化访问效率。                |

## 共享内存（Shared Memory）分块（Tiling） 优化全局内存访问

***优化步骤：分块加载→共享内存复用→结果写回***

- 从全局内存 “分块加载” 数据到共享内存
- 在共享内存中 “复用数据” 完成计算
- 将计算结果 “写回” 全局内存


**<span style="color:red;">共享内存的作用是作为高速缓存，减少对全局内存的重复访问</span>**


**<span style="color:red;">理论带宽需求与实际带宽的差距</span>**：
```markdown
浮点计算的 “内存带宽 / 浮点运算数（FLOPS）” 比例是 4B/s（即每 FLOPS 需要 4 字节内存带宽）。
Fermi GPU 的峰值浮点计算性能是 1000 GFLOPS（10 亿次浮点运算每秒）。按此计算，要达到峰值性能，需要的内存带宽是 4×1000=4000GB/s
但实际 Fermi GPU 的全局内存带宽仅约 150 GB/s，这导致性能被严重限制。按此带宽计算，实际能达到的浮点性能是 150÷4=37.5GFLOPS
```
**要接近峰值性能，必须大幅减少内存访问次数**

# Tiled Matrix Multiplication

矩阵乘法 $C=A×B$ 
> A m * n, B n * k, C m * k

**Tiled方法的核心理念**
分块矩阵乘法的基本思想：
- 将大矩阵分割成小的块(Tile)

- 将数据块加载到共享内存(Shared Memory)中

- 在共享内存中进行大部分计算

- 显著减少全局内存访问

## 朴素GPU矩阵乘法内核

```c++
__global__ void naiveMatMul(const float* A, const float* B, float* C, 
                           int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0 <= row < m
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 <= col < k
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // 每次迭代都需要从全局内存读取数据
            // A: 按行访问，但跨越n个元素 → 不连续
            // B: 按列访问 → 完全非连续，最差访问模式
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}
```

## Tiled Matrix Multiplication Kernel实现

```c++
#define TILE_SIZE 16  // 通常选择16x16的块，因为这是常见GPU的warp大小

__global__ void tiledMatMul(const float* A, const float* B, float* C, 
                           int m, int n, int k) {
    // 为每个块在共享内存中分配Tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 线程在块中的局部坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程要计算的C中的全局坐标
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有的Tile
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载A的Tile到共享内存
        int a_col = t * TILE_SIZE + tx;
        if (row < m && a_col < n) {
            As[ty][tx] = A[row * n + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // 协作加载B的Tile到共享内存  
        int b_row = t * TILE_SIZE + ty;
        if (b_row < n && col < k) {
            Bs[ty][tx] = B[b_row * k + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // 等待所有线程完成数据加载
        __syncthreads();
        
        // 从共享内存计算部分和
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        // 等待所有线程完成计算，确保共享内存不再被使用
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}
```

**内核配置与调用**

```cpp
// 内核调用示例
int m = 1024, n = 512, k = 2048;

// 定义块和网格维度
dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 16x16 = 256个线程/块
dim3 gridDim((k + TILE_SIZE - 1) / TILE_SIZE,  // 在k方向上的块数
             (m + TILE_SIZE - 1) / TILE_SIZE); // 在m方向上的块数

// 执行内核
tiledMatMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
```

## 共享内存工作关键细节
- 分块大小（TILE_SIZE）的影响
  - **需匹配共享内存容量**：例如 Fermi 架构每个线程块的共享内存约 48KB，若 TILE_SIZE=16，单个 sA 或 sB 占用 16×16×4字节（float）=1024字节，剩余空间可容纳其他数据；
  - **需避免银行冲突（Bank Conflict）**：共享内存被划分为 32 个银行（Fermi 架构），若多个线程同时访问同一银行，会导致序列化访问。通过合理设计子块索引（如行优先 / 列优先）可避免冲突。 Tips：银行总数为N，让列数 = N + 1（或其他与N互质的数），就能避免冲突
- 同步（__syncthreads()）的必要性
  - **加载后同步**：确保所有线程都已将数据写入共享内存，防止部分线程读取未就绪的数据（例如线程 0 已写完 sA[0][0]，但线程 1 尚未写完 sA[0][1]，此时线程 0 读取 sA[0][1] 会出错）；
  - **计算后同步**：防止下一轮加载子块时覆盖当前计算仍需使用的共享内存数据。
- 边界处理当矩阵尺寸 
  - N不是 TILE_SIZE 的整数倍时（如 N=1000，TILE_SIZE=16），最后一轮加载的子块会超出矩阵范围，**需将超出部分置 0**，避免越界访问全局内存。

## 支持任意矩阵维度的分块矩阵乘法核函数

1. 边界分块的正确加载是基础，通过 “条件判断 + 无效元素置 0” 确保数据访问安全；
2. 控制流的简洁性是性能保障，通过合并分支、统一逻辑减少控制分歧；
3. 功能与性能的权衡是设计核心，容忍少量无效计算以换取通用性，同时通过硬件适配将性能损失降至最低