---
layout: post
title: Heterogeneous Parallel Programming —— 笔记5
subtitle: Heterogeneous Parallel Programming
date: 2025-11-05
author: Jinz
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
  - Parallel Programming

---
# Atomic operations（原子操作）

> 在并行计算场景下，一个操作会先读取数据，对其进行修改，再将修改后的数据写回。由于多线程/多进程的并行执行，这种操作容易出现**数据竞争问题**。
>
> 当多个线程同时对同一数据执行 “读取 - 修改 - 写入” 操作时，最终结果会依赖于线程执行的先后顺序，从而产生不可预期的结果，这就是**竞态条件**。而**原子操作**的作用就是避免这种竞态条件，确保 “读取 - 修改 - 写入” 作为一个不可分割的整体执行。

*假设内存单元 Mem[x] 初始值为 0，现在有 **线程1和线程2** 同时执行以下操作*

线程 1：
Old ← Mem[x]（读取Mem[x]的值到本地变量Old）
New ← Old + 1（对Old加 1 得到New）
Mem[x] ← New（将New写回Mem[x]）

线程 2：
Old ← Mem[x]
New ← Old + 1
Mem[x] ← New

预期结果2

但实际情况会出现（**数据竞争的影响**）：

情况 1：线程 1 先完整执行，再执行线程 2
线程 1 的Old为0，New为1，写回后Mem[x]=1；
线程 2 的Old为1，New为2，写回后Mem[x]=2。

情况 2：线程 2 先完整执行，再执行线程 1
结果与情况 1 类似，最终Mem[x]=2。

情况 3：线程 1 和线程 2 的 “读取 - 修改 - 写入” 操作交叉执行
例如：
线程 1 先执行Old ← Mem[x]（Old=0）；
线程 2 执行Old ← Mem[x]（Old=0）；
线程 1 执行New ← 0+1=1，并写回Mem[x]=1；
线程 2 执行New ← 0+1=1，并写回Mem[x]=1。
此时最终Mem[x]=1，而非预期的2。

解决方案是**原子操作**（Atomic Operation） —— 指不可被中断的一个或一系列操作, 原子操作可以将 **“读取 - 修改 - 写入”** 这一系列操作**封装为一个不可分割的步骤**。

**<span style="color:red;">一个线程的操作完成前，另一个线程无法介入</span>**

## 原子操作的性能

- 原子操作的延迟与吞吐量（Latency and throughput of atomic operations）：
  
  - 延迟指执行一次原子操作所需的时间，吞吐量指单位时间内可执行的原子操作数量

- 全局内存上的原子操作（Atomic operations on global memory）：
  - 频繁的全局内存原子操作容易成为性能瓶颈
- 共享 L2 缓存上的原子操作（Atomic operations on shared L2 cache）：
  - 对共享 L2 缓存执行原子操作时，因缓存的访问速度比全局内存快，性能表现会优于全局内存上的原子操作，但仍需考虑缓存命中、竞争等因素对性能的影响
- 共享内存上的原子操作（Atomic operation on shared memory）：
  - 在共享内存上执行原子操作时，速度最快，延迟最低。合理利用共享内存上的原子操作，可有效提升涉及数据竞争场景的程序性能

## GPU中处理原子操作

1.专用硬件单元支持

GPU 中存在专门的**原子操作处理单元**，这些单元独立于通用计算核心（如 SM 中的 CUDA 核心），负责执行 “读取 - 修改 - 写入” 的原子操作流程

2.缓存层级的优化

- 共享内存原子操作：共享内存是线程块内的高速内存，其原子操作由 SM 内部的硬件直接支持，延迟极低（无需跨 SM 通信）
- L2 缓存原子操作： SM 竞争同一全局内存地址的原子操作时，L2 缓存会通过缓存一致性协议（如类似 MESI 的机制）维护数据一致性，同时利用缓存的高带宽特性减少 DRAM 访问次数，提升吞吐量。

3.线程束（Warp）级聚合优化

为减少原子操作的竞争开销，GPU 硬件或编译器会对线程束内的原子操作进行聚合：

- 当一个线程束内的多个线程对同一地址执行原子操作时，硬件会选择其中一个 “领袖线程” 来执行原子操作，其余线程的操作则在束内合并后由领袖线程统一处理。  
  > 如，32 个线程同时对全局计数器执行atomicAdd(1)，硬件会将其聚合为一次atomicAdd(32)，从而将原子操作的次数减少 32 倍

## 原子操作的执行流程

**以全局内存为例**

```markdown
1. 请求发起：线程发起原子加操作，请求被发送到对应的内存层级（如 L2 缓存或 DRAM 控制器）。
2. 独占访问：硬件对目标内存地址加锁，确保其他线程无法同时访问该地址。
3. 读取 - 修改 - 写入：读取目标地址的原始值，执行加法运算后写回，然后释放锁。
4. 结果返回：将原始值（old）返回给发起线程，完成整个原子操作。
```


![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105205423.png)


DRAM（动态随机存取存储器）上原子操作的时间延迟与串行化特性:

- 每个原子操作都包含 **DRAM 延迟（DRAM delay）**、**传输延迟（transfer delay）** 和 **内部路由延迟（internal routing）**
- 每个 “加载 - 修改 - 存储（Load-Modify-Store）” 的**原子操作会产生两次完整的内存访问延迟**，（一次读取，一次写入）

同一 DRAM 地址（即同一变量）的所有原子操作，会被强制串行化执行

**<span style="color:red;">DRAM 上原子操作的两个关键性能特点：高延迟（因两次内存访问和串行化） 和 串行执行（同一地址的原子操作无法并行）</span>**


# 私有化技术

> 直方图计算是从数据中统计元素分布的操作（例如统计图像像素强度、文本字符频率）。在并行实现中，若多个线程同时更新同一个 “直方图桶（bin）”，会引发数据竞争（race condition）—— 多个线程的 “读取 - 修改 - 写入” 操作相互干扰，导致结果错误。

早期直接使用全局内存原子操作（如atomicAdd）解决竞争，但存在严重性能问题

**“私有化” 是指为每个线程块（Thread Block）创建独立的 “私有直方图副本”，让线程先在本地（共享内存）更新私有副本，最后再合并所有副本得到最终结果**

- 共享内存访问延迟极低（仅几个时钟周期），减少原子操作开销
- 线程块内部的竞争范围缩小，大幅提升吞吐量

**进一步说明:**

- 操作需满足结合律和交换律： 私有化后的数据最终需要合并，只有当**操作（如加法）满足结合律（(a + b) + c = a + (b + c)）和交换律 a + b = b + a**时，合并的顺序才不影响最终结果

- 私有直方图的规模需较小：私有直方图通常存储在共享内存中，适配共享内存的容量

# CPU-GPU 数据传输

## 基于 DMA 的 CPU-GPU 数据传输

DMA（Direct Memory Access，直接内存访问）硬件被用于cudaMemcpy()操作以提升效率

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105210555.png)

**虚拟内存管理（Virtual Memory Management）**

1.虚拟内存机制

- 多虚拟空间映射到单物理内存：多个虚拟内存空间可以被映射到同一个物理内存上
- 虚拟地址到物理地址的转换：程序中使用的虚拟地址（如指针值）会被转换为实际的物理地址，以访问物理内存中的数据

2.内存的分页与换出

- 分页划分：每个虚拟地址空间在映射到物理内存时，会被划分为 “页（pages）”
- 内存换出（分页）：当物理内存不足时，内存页可以被 “换出（paged out）” 到外部存储（如硬盘）
- 地址转换时的存在性检查：在虚拟地址向物理地址转换的过程中，会检查某个变量是否存在于物理内存中

**数据传输与虚拟内存（Data Transfer and Virtual Memory）** 

- DMA 使用物理地址的机制
  - 执行cudaMemcpy()复制数组时，其底层是一个或多个 DMA 传输操作
  - DMA 传输的开始阶段，会进行地址转换和页存在性检查
  - 同一 DMA 传输的后续过程中，不再重复进行地址转换
- 操作系统分页机制带来的潜在问题
  - 操作系统可能会意外地将 DMA 正在读写的数据对应的内存页换出（page-out），并将另一个虚拟页 **换入（page-in）** 到同一物理地址，这会导致 DMA 传输的数据出现错误

`DMA 在虚拟内存环境下的工作特点：依赖物理地址实现高效传输，但也面临操作系统分页机制带来的潜在数据一致性风险`

## 固定内存与 DMA 数据传输

*固定内存（Pinned Memory）是被特殊标记的虚拟内存页，其特点是不能被换出（paged out）*

**固定内存与 DMA 传输的关系**:在 DMA 传输中，作为数据传输的源或目标的 CPU 内存，必须被分配为固定内存

**固定内存（Pinned Memory）的使用特性:**

1.分配的固定内存及其指针，使用方式和malloc()函数分配的内存完全相同；唯一的区别是，固定内存不会被操作系统**换出（paged out）**

2.使用固定内存时，cudaMemcpy()函数的传输速度大约能提升2倍。 避免了虚拟内存分页带来的地址转换和页存在性检查开销，让 DMA 传输更高效。

3.固定内存是有限资源，过度申请（over-subscription），会带来严重的后果（比如系统内存压力剧增、其他进程资源不足等）

`cudaHostAlloc()` 在主机上分配一块固定（锁页）内存

```cuda
cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int flags )
```
- pHost 指向已分配内存的指针的地址：用于存储分配后内存的指针地址。
- size 分配内存的大小（以字节为单位）：指定需要分配的内存字节数。
- flags 选项：目前可使用cudaHostAllocDefault作为默认选项。

`cudaFreeHost()` 释放由cudaHostAlloc()分配的固定内存

```cuda
cudaError_t cudaFreeHost ( void* ptr )
```

- ptr：输入参数。指向要释放的固定内存的指针（即 cudaHostAlloc 返回的指针）


# 任务并行（Task Parallelism）

**数据并行：**

思想：同一个任务（同一个内核）在不同的数据上执行。

CUDA实现：一个 kernel 被成千上万个线程执行，每个线程处理数据的不同部分。

例子：对一个大数组的每个元素进行平方运算。

**任务并行：**

思想：不同的任务（不同的内核或函数）可以并发执行。

CUDA实现：多个不同的 kernel 或者内存拷贝操作在同一个GPU上同时运行。

例子：一个流水线中，一个内核在进行图像模糊处理，同时另一个内核在进行边缘检测，同时还有一个DMA传输在将下一帧图像传入GPU。



**CUDA 实现任务并行的关键技术**
- 
-  流（Stream）：流 是实现任务并行的核心机制。你可以把流看作一个任务队列。

> 多个流可并行执行：不同流中的任务相互独立，GPU 会调度这些流在不同计算核心上同时运行。
> 流内任务有序：同一流中的任务严格按提交顺序执行，保证依赖关系（如先分配内存、再执行计算）。
> 典型场景：数据传输（CPU-GPU）与计算任务并行，例如用流 1 传输数据时，流 2 同时执行已传输数据的计算。  

- 默认流：所有CUDA命令（内核启动、内存拷贝）如果没有指定流，都会进入默认流。默认流中的操作是顺序执行的。

- 非默认流：你可以创建多个流。不同流中的操作可以并发执行。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1); // 创建流
cudaStreamCreate(&stream2);

// ... 使用流

cudaStreamDestroy(stream1); // 销毁流
cudaStreamDestroy(stream2);
```

## 任务并行的几种典型模式

**1.内核与内存传输并发**: GPU在执行计算时，其DMA引擎可以同时进行数据拷贝，从而隐藏数据传输的延迟。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在流1中：将数据A从主机拷贝到设备
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
// 在流2中：将数据B从主机拷贝到设备
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

// 在流1中：启动处理数据A的内核
kernel_A<<<grid, block, 0, stream1>>>(d_A);
// 在流2中：启动处理数据B的内核
kernel_B<<<grid, block, 0, stream2>>>(d_B);

// 在流1中：将结果A从设备拷贝回主机
cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream1);
// 在流2中：将结果B从设备拷贝回主机
cudaMemcpyAsync(h_B, d_B, size, cudaMemcpyDeviceToHost, stream2);
```

**2.多个内核并发**: GPU有足够的计算资源（例如多个SM），它可以同时执行来自不同流的内核

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 内核1和内核2可能会同时执行（取决于硬件资源）
kernel_Blur<<<grid1, block1, 0, stream1>>>(image_data);
kernel_EdgeDetect<<<grid2, block2, 0, stream2>>>(image_data);
```

**3.高级流水线**: 将一个大任务分解成多个阶段，每个阶段在不同的流中处理不同的数据块，形成流水线

```cpp
// 假设我们处理N个数据块
for (int i = 0; i < N; ++i) {
    int stream_id = i % 2; // 在两个流之间交替

    // 阶段1：拷贝第i块数据到设备 (流 stream[stream_id])
    cudaMemcpyAsync(d_buffer[stream_id], h_buffer[i], chunk_size, cudaMemcpyHostToDevice, stream[stream_id]);

    // 阶段2：处理第i块数据 (流 stream[stream_id])
    // 注意：这里依赖阶段1的拷贝完成，但在同一个流中，顺序是保证的。
    processing_kernel<<<grid, block, 0, stream[stream_id]>>>(d_buffer[stream_id]);

    // 阶段3：将第i块结果拷贝回主机 (流 stream[stream_id])
    cudaMemcpyAsync(h_result[i], d_buffer[stream_id], chunk_size, cudaMemcpyDeviceToHost, stream[stream_id]);
}
```

> 在这个流水线中，当流0在执行processing_kernel处理第0个数据块时，流1可能正在执行cudaMemcpyAsync将第1个数据块传入GPU


## 依赖管理与同步

**1.隐式同步**

- **默认流是一个“同步器”**。任何在默认流中的操作（例如一个同步的 cudaMemcpy 或一个没有指定流的内核启动）都会导致GPU等待所有其他流中的先前操作完成，然后自己执行，执行完成后才允许其他流继续。

最佳实践：在追求高性能的代码中，尽量避免使用默认流。

**2.显式同步**

可以让CPU或GPU等待某个流中的任务完成。

`cudaStreamSynchronize(stream)`：CPU线程 等待指定流中的所有任务完成。

`cudaDeviceSynchronize()`：CPU线程 等待所有流中的所有任务完成（不推荐，会破坏并发性）。

`cudaStreamQuery(stream)`：CPU线程 检查流中的任务是否已完成，但不会阻塞。

> 为什么会是 CPU线程等待指定流：
> 当你在CPU代码中启动一个CUDA操作（如内核或cudaMemcpyAsync）时，这个操作是异步的：
> CPU发起请求：CPU线程只是将任务"扔"到GPU的命令队列中，然后就立即继续执行后面的CPU代码。 
> GPU执行任务：GPU在后台并行地处理这些任务。
> 
> 这种"发射后不管"的模式意味着CPU线程和GPU的执行是脱节的。因此，当你需要让CPU"知道"GPU的工作进度或等待GPU完成时，你实际上是在同步CPU线程与GPU的工作状态。

**3.流内同步**

一个流内，操作是严格按发布顺序执行的。你需要担心同一个流内的内核和拷贝操作的顺序问题。

**4. 跨流同步**

*使用 CUDA Events*

```cpp
cudaEvent_t kernel1_done;
cudaEventCreate(&kernel1_done);

// 在流1中启动内核1，并在完成后记录一个事件
kernel1<<<..., stream1>>>();
cudaEventRecord(kernel1_done, stream1);

// 告诉流2，必须等待“kernel1_done”事件发生后，才能启动内核2
cudaStreamWaitEvent(stream2, kernel1_done, 0); // 0是保留标志
kernel2<<<..., stream2>>>();

cudaEventDestroy(kernel1_done);
```

> 要实现真正的任务并行，GPU硬件必须支持: **Hyper-Q（现代GPU都支持）**、**多个DMA引擎**（实现拷贝（H2D）和回拷（D2H）的并发）

- 任务并行的核心是让GPU的不同计算单元和DMA引擎同时保持忙碌。

- 流 是实现任务并行的工具，每个流是一个独立的任务队列。

- 关键模式是利用 cudaMemcpyAsync 和不同流，实现 计算与传输重叠。

- 同步是性能杀手：尽量减少全局同步（如 cudaDeviceSynchronize），使用更精细的同步方法（如事件）。

- 固定内存是前提：cudaMemcpyAsync 必须与由 cudaHostAlloc 分配的固定主机内存一起使用。

- Profile驱动优化：使用 nvprof 或 Nsight Systems 来可视化流的执行情况，确认你的任务并行策略是否真的带来了并发执行。

**创建和使用流：**

```cpp
#include <cuda_runtime.h>

int main() {
    const int N = 1024;
    int *h_a, *h_b, *d_a, *d_b;
    
    // 1. 分配固定主机内存（这是使用流和异步操作的前提！）
    cudaHostAlloc(&h_a, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_b, N * sizeof(int), cudaHostAllocDefault);
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    
    // 2. 创建两个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // 3. 在不同的流中交错安排任务
    for (int i = 0; i < 5; ++i) {
        // 在流1中：拷贝->计算->回拷 作为一个流水线阶段
        cudaMemcpyAsync(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        myKernel<<<N/256, 256, 0, stream1>>>(d_a); // 注意最后一个参数指定流
        cudaMemcpyAsync(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        
        // 在流2中：同时进行另一个流水线阶段
        cudaMemcpyAsync(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice, stream2);
        myKernel<<<N/256, 256, 0, stream2>>>(d_b);
        cudaMemcpyAsync(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    }
    
    // 4. 同步与清理
    cudaStreamSynchronize(stream1); // CPU等待stream1完成
    cudaStreamSynchronize(stream2); // CPU等待stream2完成
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
```

**<span style="color:red;">一个线程的操作完成前，另一个线程无法介入</span>**