---
layout: post
title: Heterogeneous Parallel Programming —— 笔记3
subtitle: Heterogeneous Parallel Programming
date: 2025-11-03
author: Jinz
header-img: img/post-bg5.jpg
catalog: true
tags:
  - Parallel Programming

---

# Performance Considerations DRAM Bandwidth

**DRAM 带宽指的是 CPU（或 GPU）与内存（DRAM）之间每秒能传输的数据量，单位通常是 GB/s，它直接决定了处理器获取数据的速度，进而影响整体系统性能**

`带宽 = 内存频率 × 位宽 × 通道数 / 8（除以 8 是将 bit 转换为 Byte）`

`例如： DDR5-6400 双通道内存的理论带宽为：6400MHz × (64bit×2) / 8 = 102.4GB/s`

## DRAM Bank（存储体）的组织架构
![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103105716.png)

DRAM Bank 的逻辑架构:

- Row Decoder（行解码器）
  - 接收行地址（Row Addr） 后，定位到内存核心阵列中对应的 “行”。DRAM 采用 “行 - 列” 二维寻址方式，行解码器是第一步寻址的关键
- Memory Cell Core Array（存储单元核心阵列）
  - 实际存储数据的 “矩阵”，包含大量行和列的存储单元（“电容器 + 晶体管” 结构）。行解码器选中某一行后，这一行的所有位会被临时传输到Sense Amps（灵敏放大器）
- Sense Amps（灵敏放大器）
  - 放大信号，并“锁存” 行数据
- Column Latches（列锁存器）
  - 对灵敏放大器输出的整行数据进行暂存
- Mux（多路选择器）
  - 接收**列地址（Column Addr）** 后，**从列锁存器的 “宽（Wide）” 数据中，选择需要的 “窄（Narrow）” 数据通道**，最终通过Pin Interface（引脚接口） 输出到芯片外部（即Off-chip Data，芯片外数据）

> **带宽**：Mux 的 “宽→窄” 转换、引脚接口的速率，以及 “行 - 列” 寻址的并行度（多个 Bank 同时操作），共同决定了数据传输的总带宽。
**延迟**：行激活、列选择的过程存在固有延迟（如 “行缓冲命中”“行冲突” 等场景），这也是 DRAM 延迟高于 CPU 缓存的主要原因

## DRAM 核心阵列（Core Array）的速度瓶颈

**DRAM 的核心存储阵列（即存储数据的物理单元区域）是 “慢” 的，这是 DRAM 延迟和带宽受限的底层原因之一**

不同代际 DRAM 的 “核心速度” 与 “接口速度” 的比例：

DDR：核心速度 = 接口速度的 1/2

DDR2/GDDR3：核心速度 = 接口速度的 1/4

DDR3/GDDR4：核心速度 = 接口速度的 1/8

趋势：未来这一差距可能会进一步拉大（“… likely to be worse in the future”）

*“接口速度” 指 DRAM 与外部（如 CPU、GPU）通信的引脚速率，“核心速度” 指内部存储阵列的读写速率*

> 代际发展中，核心速度与接口速度的差距越来越大；
存储单元的物理设计（小电容、多单元共享线路）导致其读写天然缓慢。

## DRAM 突发传输（DRAM Bursting）

**DRAM 提升带宽的关键技术之一，核心是通过 “批量读取 + 分步骤传输” 来弥补核心阵列速度与接口速度的差距**

1. 批量加载数据到内部缓冲
  一次性从同一行（same row）中读取 N × 接口位宽 的 DRAM 比特（bits），并将这些数据加载到内部缓冲区。
1. 以接口速度分步骤传输
  然后以接口速度，分 N 个步骤将缓冲区中的数据传输到外部。

例如：对于 DDR2 或 GDDR3，N=4（即核心速度是接口速度的 1/4），因此：
缓冲区宽度 = 4 × 接口位宽，也就是说，内部缓冲区的宽度是接口位宽的 4 倍，以此实现 “一次加载 4 倍接口宽度的数据，再分 4 步以接口速度传输”，从而提升整体带宽

**先攒一波数据，再高速分批送走**

# Memory Coalescing（内存合并）

**内存合并 是指当同一个 Warp（32个线程）中的线程访问连续的、对齐的全局内存地址时，GPU 可以将这些多个内存访问合并为少数几个内存事务（memory transactions），从而大幅提高内存带宽利用率**

1. **地址对齐**
   
2. **连续且紧凑的访问模式**

> 合并访问：线程束 32 个线程访问float数组arr[0...31]，地址连续且对齐，GPU 仅需 1 次内存事务完成传输，带宽利用率接近 100%。
非合并访问：线程束线程访问arr[0], arr[2], arr[4]...（地址不连续），GPU 可能拆分为 16 次内存事务，带宽利用率骤降，性能差距可达数倍。


# Parallel Computation Patterns Convolution(卷积并行计算模式)

**卷积的计算逻辑（以 2D 卷积为例）:** 2D 卷积是 “输入特征图（Input Feature Map）” 与 “卷积核（Kernel）” 的滑动窗口计算，核心是逐位置的 “元素相乘 + 求和”


**带边界条件处理的 1D 卷积 CUDA 内核**

```cpp
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; j++) {
      if (N_start_point + j >= 0 && N_start_point + j < Width) {
        Pvalue += N[N_start_point + j]*M[j];
      }
    }
    P[i] = Pvalue;
}
```

核心是 **“每个线程计算一个输出元素”**，并通过if判断处理边界情况（超出输入范围的元素记为 0）。它是 CUDA 中实现 1D 卷积的基础模板，体现了 “线程与输出元素一一对应” 的并行模式，同时通过边界判断保证计算的正确性。

## 线程到 输出/输入 数据索引的映射

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103115242.png)

**线程到输出数据索引的映射：** 用于明确每个线程块、每个线程负责计算输出数据的哪个部分。数组P是输出数据，被划分为多个 “块（tile）”，每个块的宽度为O_TILE_WIDTH

**<span style="color:red;">“任务分配与数据索引管理” 的核心逻辑</span>**

**每个线程块（Thread Block）负责计算一个输出块（Output Tile）**

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103115447.png)

**线程到输入数据的索引映射：** 用于明确 CUDA 线程在处理卷积等操作时，如何根据输出数据的索引定位到需要读取的输入数据区域

## Parallel Computation Patterns Tiled Convolution


分块卷积的核心是 “以块为单位处理数据，利用共享内存缓存输入数据的局部块，减少全局内存访问次数”，具体流程如下（以 2D 卷积为例）：

1. 数据分块（Tiling）

输出分块（Output Tile）：将输出特征图划分为多个尺寸为 O_TILE_HEIGHT × O_TILE_WIDTH 的块，每个线程块负责计算一个输出块。

输入分块（Input Tile）：每个输出块的计算依赖输入特征图的一个更大的局部区域（需覆盖卷积核滑动时涉及的所有输入元素），称为输入块，尺寸为 I_TILE_HEIGHT × I_TILE_WIDTH，计算公式为：

**`I_TILE_HEIGHT = O_TILE_HEIGHT + KERNEL_HEIGHT - 1`**

**`I_TILE_WIDTH = O_TILE_WIDTH + KERNEL_WIDTH - 1`**

（KERNEL_HEIGHT/WIDTH 为卷积核的高 / 宽，多出的部分是为了覆盖卷积核边缘的输入数据）。

2. 共享内存复用

线程块先将输入块从全局内存加载到共享内存（Shared Memory）中（共享内存是 GPU 片上内存，延迟仅为全局内存的 1/100 左右）。

线程块内的线程计算输出块时，直接从共享内存读取输入数据，而非重复访问全局内存。

3. 线程分工

线程块内的每个线程负责计算输出块中的一个元素（或多个相邻元素）。

计算时，线程根据自身在块内的索引，从共享内存中读取对应的输入窗口和卷积核数据，执行 “乘加求和” 操作。

**优点**：

**<span style="color:red;">减少全局内存访问</span>**

**<span style="color:red;">提升内存访问效率</span>**

**<span style="color:red;">适配硬件缓存</span>**

- 确定分块大小：根据卷积核尺寸、共享内存容量和线程块大小（如 16×16 线程），设置输出块大小 O_TILE = 16×16，计算输入块大小 I_TILE = (16 + K-1)×(16 + K-1)（K 为卷积核尺寸）。

- 加载输入块到共享内存：线程块内的线程分工读取输入特征图的对应区域，按合并访问方式写入共享内存（如 s_Input[ty][tx] = N[global_row][global_col]，ty/tx 为线程块内的二维索引）。

- 同步线程块：用 __syncthreads() 确保所有线程完成共享内存加载，避免数据未就绪导致的错误。

- 计算输出块：每个线程根据自身索引，从共享内存读取 K×K 的输入窗口和卷积核，计算输出元素并写入全局内存。

- 处理边界条件：对于边缘的输入块（超出输入特征图范围的部分），通过条件判断设置为 0 或其他边界值（如填充 0）。

> 非分块卷积：计算 256×256 输出，3×3 卷积核，每个输出元素需读取 9 个输入元素，总全局内存访问约 256×256×9 = 589,824 次。
分块卷积（16×16 输出块）：输入块大小 18×18（16+3-1），每个块加载 18×18=324 次，共 (256/16)×(256/16)=256 个块，总访问 256×324=82,944 次，仅为非分块的 14%

##  内存访问优化

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103120804.png)

*自动填充是内存访问优化的关键手段：DRAM 的突发传输是 “按块” 进行的（如一次传输 128 字节），如果矩阵行长度不是突发粒度的整数倍，内存访问会出现 “碎片化”，导致带宽利用率下降。通过填充让行长度对齐到突发边界，可确保每次内存访问都能 “满额传输”，最大化 DRAM 带宽的利用率*

## 利用常量内存（Constant Memory）和缓存来优化卷积核（Mask）的访问性能

> 卷积核在卷积运算中被所有线程读取，但不会被修改（属于只读数据）
> 
> 同一线程束（Warp）内的所有线程会在同一时间访问卷积核的相同位置（因为卷积核是全局共享的权重模板）

**常量内存（Constant Memory），其内容会被 “激进地缓存”**

- 缓存的值会广播到线程束内的所有线程：即线程束中只需有一个线程读取常量内存中的某个值，其他线程可直接复用该缓存值，无需重复访问内存。

- 效果：在不消耗共享内存（Shared Memory）的前提下，等效提升内存带宽（因为大量线程的访问被 “合并” 为一次缓存读取）。

在卷积核参数中使用 `const __restrict__` 限定符，告知编译器该数据可被常量缓存优化

```cpp
__global__ void convolution_2D_kernel(float *P, float *N, height, width, channels, const float __restrict__ *M) {
    // 内核逻辑
}
```

> - **const** 表示 M（卷积核）是只读的；
>
> - **`__restrict__`** 表示 M 没有别名（即不会被其他指针同时访问），编译器可安全地对其应用常量缓存优化

*针对卷积核 “多线程只读、访问模式一致” 的特点，将原本可能产生的大量重复内存访问，转化为高效的缓存广播*

## 数据复用（Data Reuse）

*目标是通过重复利用已加载到高速存储（如 GPU 共享内存）中的数据，减少对高延迟全局内存的访问次数*

核心原理:  分块卷积中，输入数据的局部区域会被多个输出元素重复使用（卷积核滑动时，相邻输出元素的输入窗口高度重叠）

数据复用的逻辑是：**将这些重叠的输入数据一次性加载到共享内存（或其他高速存储）中**，供多个线程重复读取，避免对全局内存的多次重复访问。

**<span style="color:red;">利用输入数据的局部重叠性，将高延迟的全局内存访问转化为低延迟的片上存储（共享内存、常量缓存）访问</span>**

**实现方式**

1. 输入数据的分块加载与复用

- 分块加载输入数据到共享内存

线程块将当前输出块对应的输入块（尺寸为 `I_TILE = O_TILE + K - 1`，其中 `O_TILE` 是输出块尺寸，`K` 是卷积核尺寸）从全局内存加载到共享内存。

```cpp
__shared__ float s_N[I_TILE_HEIGHT][I_TILE_WIDTH];
int tx = threadIdx.x, ty = threadIdx.y;
int global_x = blockIdx.x * O_TILE_WIDTH + tx;
int global_y = blockIdx.y * O_TILE_HEIGHT + ty;

// 加载输入块到共享内存
s_N[ty][tx] = N[global_y * width + global_x];  

// 同步确保所有线程加载完成
__syncthreads(); 
```

- 线程块内复用共享内存数据

线程块内的每个线程计算输出块中的一个元素时，直接从共享内存中读取输入窗口数据，而非再次访问全局内存。例如，计算输出元素 (i,j) 时，从共享内存中读取以 (i,j) 为中心的 K×K 输入窗口：

```cpp
float val = 0;
for (int k = 0; k < K; k++) {
    for (int l = 0; l < K; l++) {
        val += s_N[ty + k][tx + l] * M[k * K + l];  // 复用共享内存数据
    }
}
P[global_y * width + global_x] = val;
```

2.  卷积核的复用（结合常量内存）

将卷积核存入常量内存后，同一线程束（Warp）的所有线程访问同一卷积核元素时，硬件会自动广播缓存的数值，避免重复读取全局内存

## Bandwidth Reduction（带宽缩减）

*通过优化内存访问模式，减少对全局内存（或其他高延迟存储）的总数据传输量，从而降低对内存带宽的依赖，缓解带宽瓶颈*

**全局内存的带宽是并行计算（如 GPU）的核心瓶颈之一：**

**每次访问全局内存需要传输固定大小的数据块（如 DRAM 的突发传输粒度），且延迟高；**

**若大量数据被重复访问（如卷积中的输入窗口重叠），会导致 “无效带宽消耗”（同一数据被多次传输）**


**<span style="color:red;">实现带宽缩减的关键手段:</span>**

- **<span style="color:blue;">数据复用</span>**

- **<span style="color:blue;">分块与局部性优化</span>**

- **<span style="color:blue;">内存访问合并</span>**

- **<span style="color:blue;">常量 / 纹理内存缓存</span>**


1. 公式推导

$带宽缩减比例 = 1− 优化前总传输量/优化后总传输量$
​
2. 示例（分块卷积场景）

优化前（无数据复用）：输出块大小 O=16×16，卷积核大小 K=3×3，输入数据类型为float（4 字节）。

总传输量 = O×K×4=16×16×9×4=9216 字节。

优化后（分块 + 数据复用）：输入块大小 I=(16+3−1)×(16+3−1)=18×18。

总传输量 = I×4=18×18×4=1296 字节。

带宽缩减比例：1− 9216/1296 ≈86%，即带宽需求降低了 86%


![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251103123047.png)

**通过数据复用，原本需要传输 MASK_WIDTH × O_TILE_WIDTH 个数据，优化后仅需传输 O_TILE_WIDTH + MASK_WIDTH - 1 个数据，两者的比值即为带宽缩减的 “收益倍数”—— 比值越大，带宽缩减效果越显著**