---
layout: post
title: Introduction to Parallel Computing
subtitle: Basic concepts and introduction knowledge
date: 2024-08-26
author: Jinz
header-img: img/post-bg5.jpg
catalog: true
tags:
  - Parallel Computing
---

# 概念

*what is parallel compution?*
并行计算是指**同时使用多个计算资源来解决一个计算问题**的方法；

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261520310.gif)

*why use parallel compution?*
节约时间与金钱，解决更复杂的问题，提供并发性，利用非本地资源，更好的利用底层硬件；

*who is using parallel compution?*
Science and Engineering, Industrial and Commercial, Global Applications

## 并行计算机分类

弗林把多处理器计算机分为以下四类

![Flynn's taxonomy 弗林的分类法](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261518334.gif)

**其中SISD是一种串行计算机，SIMD（GPU上）和MIMD（超算上）使用最多**

# Parallel Computer Memory Architectures——并行计算机内存架构

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261533542.png)

## Shared Memory——共享内存

- 具有**所有处理器访问所有内存作为全局地址空间**的能力
- 多处理器能够独立运行并访问相同的内存资源
- 一个处理器对内存的改变对于所有处理器都可见

根据访问时间可以分为 *UMA* 和 *UNMA*

### Uniform Memory Access (UMA) —— 统一内存访问 

常见的是 *Symmetric Multiprocessor (SMP)* 对称多处理器， **对内存的访问时间相同**

- 有时也称为 *CC-UMA - 缓存一致性 UMA* 
    
    > 缓存一致性意味着如果一个处理器更新共享内存中的某个位置，所有其他处理器都会知道该更新。缓存一致性是在**硬件级别**实现的。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261544606.gif)

### Non-Uniform Memory Access (NUMA) —— 非统一内存访问

通常通过物理连接两个或多个SMP，一个SMP可以直接访问另一个SMP的内存

- 跨链接的内存访问速度较慢
- 如果保持缓存一致性，则可以称为 *CC-NUMA*

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261550504.gif)

## Distributed Memory——分布式内存

- 通过网络连接处理器之间的内存
- 处理器有自己的本地内存。一个处理器中的内存地址不会映射到另一处理器
  > **不存在跨所有处理器的全局地址空间**的概念
  **缓存一致性的概念不适用**

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261601245.gif)

## Hybrid Distributed-Shared Memory——混合分布式共享内存

- 同时采用共享和分布式内存架构
- 共享存储器组件可以是共享存储器机器和/或图形处理单元(GPU)
- 机器只知道自己的内存，而不知道另一台机器上的内存。因此，需要网络通信将数据从一台机器移动到另一台机器。
  
![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261605573.gif)

# Parallel Programming Models——并行编程模型

常用的并行编程模型有以下几种：

- Shared Memory (without threads) —— 共享内存（无线程）
- Threads —— 线程式
- Distributed Memory / Message Passing —— 分布式内存/消息传递
- Data Parallel —— 数据并行
- Hybrid —— 混合式
- Single Program Multiple Data (SPMD) —— 单程序多数据 (SPMD)
- Multiple Program Multiple Data (MPMD) —— 多程序多数据 (MPMD)

## Shared Memory (without threads) —— 共享内存（无线程）

在此编程模型中，进程/任务共享一个公共地址空间，它们异步读取和写入该地址空间。

>锁/信号量之类的各种机制用于控制对共享内存的访问、解决争用并防止竞争条件和死锁

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261927189.gif)

## Threads Model —— 线程模型

这种编程模型是一种共享内存编程，并行编程的线程模型中，单个“重量级”进程可以具有多个“轻量级”并发执行路径


例子：

1. 主程序a.out被安排由本机操作系统运行。 a.out加载并获取运行所需的所有系统和用户资源。这就是“重量级”的过程。
2. a.out执行一些串行工作，然后创建许多可以由操作系统同时调度和运行的任务（线程）。
3. 每个线程都有本地数据，但也共享a.out的整个资源。这节省了与为每个线程复制程序资源相关的开销（“轻量级”）。每个线程还受益于全局内存视图，因为它共享a.out的内存空间。
4. 线程的工作最好被描述为主程序中的子例程。任何线程都可以与其他线程同时执行任何子例程。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261930445.gif)

## Distributed Memory / Message Passing —— 分布式内存/消息传递

- 在计算期间使用自己的本地内存的一组任务。多个任务可以驻留在同一物理机器上和/或跨任意数量的机器。
- 任务通过发送和接收消息来进行通信来交换数据。
- 数据传输通常需要各个进程协同操作，发送操作必须有匹配的接收操作。
  
![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261934904.gif)

## Data Parallel Model —— 数据并行模型

- 地址空间被全局对待
- 大多数并行工作都集中在对数据集执行操作。数据集通常被组织成通用结构，例如数组或立方体。
- 一组任务共同作用于同一数据结构，但是每个任务作用于同一数据结构的不同分区。
- 任务对其工作分区执行相同的操作，例如“向每个数组元素添加 4”。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261936515.gif)

## Hybrid Model —— 混合模型

**常见的模型如下**：

### Hybrid model with MPI and OpenMP:

- 线程使用本地节点数据执行计算密集型内核
- 不同节点上的进程之间的通信使用 MPI 通过网络进行

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261937915.gif)

### Hybrid model with MPI and CUDA

- MPI 任务使用本地内存在 CPU 上运行，并通过网络相互通信。
- 计算密集型内核被卸载到节点上的 GPU。
- 节点本地内存和 GPU 之间的数据交换使用 CUDA（或等效的东西）。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261939697.gif)

## Single Program Multiple Data (SPMD) —— 单程序多数据 (SPMD)

*SPMD 实际上是一种“高级”编程模型，可以构建在前面提到的并行编程模型的任意组合之上*

- 单个程序：所有任务同时执行同一程序的副本。
- 多个数据：所有任务可能使用不同的数据
- 任务不一定必须执行整个程序 - 也许只是其中的一部分。
- SPMD 模型使用消息传递或混合编程，可能是多节点集群最常用的并行编程模型。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261943166.gif)

## Multiple Program Multiple Data (MPMD) —— 多程序多数据 (MPMD)

- 多个程序：任务可以同时执行不同的程序。
- 多个数据：所有任务可能使用不同的数据

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261944955.gif)

# Designing Parallel Programs —— 设计并行程序

***Questions to ask: 
Is this problem able to be parallelized?
这个问题可以并行化吗？
How would the problem be partitioned?
问题将如何划分？
Are communications needed?
是否需要沟通？
Are there any data dependencies?
是否存在数据依赖性？
Are there synchronization needs?
有同步需求吗？
Will load balancing be a concern?
负载平衡会成为一个问题吗？***

## 1. Understand the Problem and the Program 

## 2. Partitioning 分区

**域分解和功能分解**

### Domain Decomposition 域分解

在这种类型的分区中，与问题相关的数据被分解。然后，每个并行任务都处理一部分数据。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261949939.gif)

### Functional Decomposition 功能分解

在这种方法中，重点是要执行的计算，而不是计算所操纵的数据。问题根据必须完成的工作进行分解。然后，每个任务执行整体工作的一部分。

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261950685.gif)

## 3. Communications 通讯

## 4. Synchronization 同步

## 5. Data Dependencies 数据依赖性

当语句执行顺序影响程序结果时，程序语句之间就存在依赖性。

>数据依赖性是由不同任务多次使用存储中的相同位置造成的。

## 6. Load Balancing 负载均衡

负载平衡是指在任务之间分配大致相等的工作量，以便所有任务始终保持忙碌的做法。可以认为是任务空闲时间的最小化。


## 7. Granularity 粒度

在并行计算中，粒度是计算与通信比率的定性度量。

Fine-grain parallelism 细粒度并行性
- 通信事件之间完成相对少量的计算工作。
- 有利于负载平衡。
- 意味着较高的通信开销和较少的性能增强机会。
- 


![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261954065.gif)

Coarse-grain parallelism 粗粒度并行性
- 通信/同步事件之间完成相对大量的计算工作
- 意味着更多的绩效提升机会
- 更难有效地进行负载平衡

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408261955079.gif)

> 在大多数情况下，与通信和同步相关的开销相对于执行速度而言较高，因此具有粗粒度是有利的。

## 8. I/O 输入/输出

## 9. Debugging 调试

## 10. Performance Analysis and Tuning 性能分析与调优

# 参考
1. [并行计算入门](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial##DesignDependencies)