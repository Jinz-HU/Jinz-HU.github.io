---
layout: post
title: Heterogeneous Parallel Programming —— 笔记1
subtitle: Heterogeneous Parallel Programming
date: 2025-10-30
author: Jinz
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
  - Parallel Programming

---

# CPU GPU

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030191048.png)

**CPU “延迟导向设计”（Latency Oriented Design）** 
降低 “延迟”（即操作或数据等待的时间），从而提升执行效率

* **Powerful ALU（强大的算术逻辑单元）**：直接减少单条指令的执行延迟
* **Large caches（大容量缓存）**:CPU 缓存是介于 CPU 和主存之间的高速存储层,把需要频繁访问的数据临时存到缓存中，从而将 “访问主存的长延迟” 转化为 “访问缓存的短延迟”
* **Sophisticated control（复杂的控制逻辑）**:Branch prediction（分支预测）提前预判分支走向，减少因等待分支结果导致的延迟;Data forwarding（数据前推）直接把未写回的中间结果传递给下一条指令，减少数据等待。


![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030191700.png)
**GPU “吞吐量导向设计”（Throughput Oriented Design）**
通过并行处理提升 “吞吐量”（单位时间内完成的任务总量），而非降低单任务的延迟

* **Small caches（小容量缓存）**：减少数据在缓存中的冗余存储，让更多数据通过内存总线并行传输，从而提升整体内存吞吐量（单位时间内传输的数据量）
* **Simple control（简单的控制逻辑）**： 放弃对单线程延迟的优化，转而通过大规模并行 “掩盖延迟”（让其他线程在等待时继续工作）
* **Energy efficient ALUs（高能效的算术逻辑单元）**：ALU 通过深度流水线设计，能同时处理大量并行任务（每个流水线阶段处理不同任务的一部分），最终在单位时间内完成更多运算，即高吞吐量
* **Require massive number of threads to tolerate latencies（需要海量线程来容忍延迟）**：GPU 不试图降低延迟，而是通过 “延迟容忍” 来维持高吞吐量；当部分线程因内存访问或运算延迟等待时，GPU 可以快速切换到其他就绪线程继续执行。只要线程数量足够多（通常数万到数百万），就能保证硬件资源（如 ALU、内存总线）始终被充分利用，从而抵消单线程延迟带来的影响，维持整体高吞吐量

![ ** 冯・诺依曼架构处理器（Von-Neumann Processor）** ](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030192501.png)
* Memory（内存）
* Processing Unit（处理单元）
  * ALU（算术逻辑单元）
  * Reg File（寄存器文件）
* Control Unit（控制单元）
  * PC（程序计数器）
  * IR（指令寄存器）
* I/O（输入 / 输出设备）

```
取指：控制单元通过 PC 获取下一条指令的地址，从内存中读取指令并存储到 IR 中，同时 PC 自动递增（指向下一条指令）。
解码：控制单元解析 IR 中的指令，确定要执行的操作（如加法、数据移动等）和操作数来源（内存地址或寄存器）。
执行：若涉及数据运算，控制单元协调 ALU 从寄存器或内存中读取数据，完成运算后将结果写回寄存器或内存。
      若涉及 I/O 或跳转，控制单元协调对应的硬件模块执行操作（如修改 PC 实现跳转）。
循环：重复 “取指 - 解码 - 执行” 的过程，直到程序执行完毕。
```

# CUDA 异构计算

## CUDA 异构计算中向量加法（vecAdd）的主机端代码流程

```c++
#include <cuda.h>
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float);
    float* d_A, d_B, d_C;

    1. // 分配设备内存 & 复制A、B到设备内存
    2. // 启动内核：GPU执行实际的向量加法
    3. // 将C从设备内存复制回主机内存
}
```

**设备内存管理与数据上传**：
调用cudaMalloc为d_A, d_B, d_C分配 GPU 可访问的设备内存。
调用cudaMemcpy将h_A, h_B的数据从主机内存复制到d_A, d_B的设备内存中。
**内核启动**：
编写并启动 CUDA 内核（如vecAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n)），让 GPU 的多个线程并行执行向量加法（每个线程处理一个元素：d_C[i] = d_A[i] + d_B[i]）。
**结果下载**：
调用cudaMemcpy将d_C的结果从设备内存复制回主机内存的h_C中，完成整个计算流程。

**cudaMalloc()：GPU 设备内存分配**
```
float* d_A;
int n = 1000;
cudaMalloc(&d_A, n * sizeof(float)); // 为1000个float元素分配设备内存
```
**cudaFree()：GPU 设备内存释放**
```
cudaFree(d_A); // 释放d_A指向的GPU设备内存
```
**cudaMemcpy()：主机与设备间的数据传输**
`cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
> dst：目标内存指针（可以是主机或设备内存）。
src：源内存指针（可以是主机或设备内存）。
count：要复制的字节数。
kind：传输方向，常用取值：
cudaMemcpyHostToDevice：从主机内存复制到设备内存。
cudaMemcpyDeviceToHost：从设备内存复制到主机内存。
cudaMemcpyDeviceToDevice：在设备内存内部复制。

```c++
float* h_A = (float*)malloc(n * sizeof(float)); // 主机内存分配
float* d_A;
cudaMalloc(&d_A, n * sizeof(float));

// 主机→设备：将h_A的数据复制到d_A
cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);

// 设备→主机：将d_A的计算结果复制回h_C
float* h_C = (float*)malloc(n * sizeof(float));
cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
```

三者协作流程：
1. 用cudaMalloc()在 GPU 上分配设备内存。
2. 用cudaMemcpy()（HostToDevice）将 CPU 数据复制到 GPU 设备内存。
3. 启动 GPU 内核，在设备内存上执行计算。
4. 用cudaMemcpy()（DeviceToHost）将 GPU 计算结果复制回 CPU 内存。
5. 用cudaFree()释放 GPU 设备内存，完成资源回收。



## CUDA 中线程块（Thread Block）并行执行向量加法的核心逻辑

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030193907.png)

每个线程通过公式 i = blockIdx.x * blockDim.x + threadIdx.x 计算自己负责处理的全局索引i,实现 “一个线程处理一个元素” 的并行逻辑

1. Device code（GPU 端代码）的能力
**R/W per-thread registers**：
可以读写每个线程独有的寄存器。寄存器是 GPU 线程的高速私有存储，每个线程有自己的寄存器空间，用于暂存运算的中间数据，访问速度极快。
**R/W all-shared global memory**：
可以读写所有线程共享的全局内存。全局内存是 GPU 上的大容量内存区域，所有线程（无论属于哪个线程块）都可以访问，但访问延迟高于寄存器和共享内存。
2. Host code（CPU 端代码）的能力
**Transfer data to/from per grid global memory**：
可以与整个网格（Grid）共享的全局内存进行数据传输。主机端通过 CUDA 的 API（如cudaMemcpy），将数据从 CPU 内存复制到 GPU 的全局内存，或从 GPU 全局内存复制回 CPU 内存，以此实现 CPU 和 GPU 之间的数据交互。

# CUDA 向量加法完整流程

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030194945.png)

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030195023.png)

一、主机端控制代码（Vector Addition Kernel Host Code）
功能：在 CPU 上完成 GPU 内存管理、数据传输和内核启动的控制逻辑。
代码解析：
函数vecAdd：封装了向量加法的完整流程（内存分配、数据传输、内核启动等步骤在此处省略了细节）。
内核启动：vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
<<<gridDim, blockDim>>>：CUDA 特有语法，用于指定启动的线程块数量（gridDim）和每个块的线程数量（blockDim）。
ceil(n/256.0)：根据向量长度n计算所需的线程块数量（向上取整，确保所有元素都被处理）。
d_A, d_B, d_C：指向 GPU 设备内存的指针，用于传递数据给内核。

二、设备端内核代码（Vector Addition Kernel）
功能：在 GPU 上并行执行向量加法（C = A + B），每个线程负责一个元素的加法运算。
代码解析：
`__global__`：标识这是一个可由 CPU 调用、在 GPU 上执行的内核函数。
线程索引计算：int i = threadIdx.x + blockDim.x * blockIdx.x;
threadIdx.x：线程在其所属线程块内的索引。
blockDim.x：每个线程块的线程数量（此处隐含为 256）。
blockIdx.x：线程块在网格中的索引。
该公式将每个线程唯一映射到向量的一个元素位置i。
边界检查与计算：if (i < n) C[i] = A[i] + B[i]; 确保线程仅处理有效范围内的元素（i < 向量长度n），并执行对应位置的加法。

三、整体流程总结
主机端准备：CPU 通过cudaMalloc分配 GPU 设备内存，再通过cudaMemcpy将输入向量A、B从主机内存复制到设备内存。
GPU 并行计算：CPU 调用vecAddKernel内核，GPU 启动大量线程并行执行向量加法。
结果回传：CPU 通过cudaMemcpy将 GPU 计算得到的向量C从设备内存复制回主机内存，完成整个计算流程。

**串行矩阵乘法**
![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030195359.png)


**CUDA 并行计算矩阵乘法**
```c++
__global__ void MatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < k)) {
        float Cvalue = 0.0;
        for (int i = 0; i < n; ++i)
            /* A[Row, i] and B[i, Col] */
            Cvalue += A[Row * n + i] * B[Col + i * k];
        C[Row * k + Col] = Cvalue;
    }
}
```

## CUDA 函数声明类型

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251030195805.png)
 