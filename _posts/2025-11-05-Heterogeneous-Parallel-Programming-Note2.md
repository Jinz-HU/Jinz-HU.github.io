---
layout: post
title: Heterogeneous Parallel Programming —— 笔记6
subtitle: Heterogeneous Parallel Programming
date: 2025-11-05
author: Jinz
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
  - Parallel Programming

---

# OpenCL

> OpenCL 是一个开放的、跨平台的并行编程框架。它的全称是 Open Computing Language
>
> 支持来自不同厂商的GPU、CPU，甚至其他专用芯片,可以运行在 NVIDIA/AMD/Intel 的 GPU 以及 CPU、FPGA 等多种设备上

| OpenCL 概念           | CUDA 等效概念 | 详细解释                                                                                                                                 |
| --------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| host                  | host          | 指传统的 CPU 及其内存环境，负责管理设备、创建上下文、编译内核、分派任务等控制工作。                                                      |
| device                | device        | 指执行并行计算的硬件，通常是 GPU，也可以是 CPU、FPGA 等其他加速器。                                                                      |
| kernel                | kernel        | 在设备上并行执行的函数。是用 OpenCL C 或 CUDA C 编写的核心算法。                                                                         |
| host program          | host program  | 运行在主机上的程序，负责设置执行环境、管理内存、启动内核等任务。                                                                         |
| NDRange (index space) | grid          | 定义了所有并行执行实例的整体组织结构。可以是一维、二维或三维的索引空间。                                                                 |
| work item             | thread        | 数据并行的最小执行单位。每个 work item/thread 执行相同的内核代码，但处理不同的数据。                                                     |
| work group            | block         | 工作项/线程的分组。组内的 work item/thread 可以：<br>• 访问快速的本地内存/共享内存<br>• 使用屏障同步进行协调<br>• 在同一个计算单元上执行 |



![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105225632.png)


- 图展示了一个 一维 NDRange，它被分成了多个工作组：

```text
NDRange 结构：
[Work-Group 0] [Work-Group 1] ... [Work-Group 7]
[0-7]          [8-15]          ... [56-63]
```

- 单一程序多数据

```text
int id = get_global_id(0);
result[id] = a[id] + b[id];
```

这段代码在每个工作项中执行，但行为不同：

Work item 0: id=0 → result[0] = a[0] + b[0]

Work item 9: id=9 → result[9] = a[9] + b[9]

Work item 15: id=15 → result[15] = a[15] + b[15]


`SPMD`: 所有工作项执行完全相同的内核代码，每个工作项通过 get_global_id() 获取唯一ID，从而处理不同的数据元素

```text
时间点 t0:
GPU Core 0: 执行 work item 0 → result[0] = a[0] + b[0]
GPU Core 1: 执行 work item 1 → result[1] = a[1] + b[1]
GPU Core 2: 执行 work item 2 → result[2] = a[2] + b[2]
...
GPU Core 7: 执行 work item 7 → result[7] = a[7] + b[7]

时间点 t1:
GPU Core 0: 执行 work item 8 → result[8] = a[8] + b[8]
...
```

OpenCL 和 CUDA 的工作项标识和查询函数
- 
![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105225953.png)
---

OpenCL
-  
```cpp
__kernel void vector_add(__global float* a, 
                         __global float* b, 
                         __global float* result) {
    int global_id = get_global_id(0);      // 全局索引
    int local_id = get_local_id(0);        // 组内索引
    int global_size = get_global_size(0);  // 总工作项数
    int local_size = get_local_size(0);    // 工作组大小
    
    // 边界检查
    if (global_id < global_size) {
        result[global_id] = a[global_id] + b[global_id];
    }
}
```

Cuda 等效实现
- 

```cpp
__global__ void vector_add(float* a, float* b, float* result) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局索引
    int local_id = threadIdx.x;                             // 块内索引
    int global_size = gridDim.x * blockDim.x;               // 计算总线程数
    int local_size = blockDim.x;                            // 块大小
    
    // 边界检查
    if (global_id < global_size) {
        result[global_id] = a[global_id] + b[global_id];
    }
}
```

## OpenCL设备架构

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105230638.png)

```text
设备架构层次       执行模型对应
============      =============
Device            ─── NDRange
  │
  ├─ Compute Unit ─── Work-Group
  │     │
  │     ├─ Processing Element ─── Work-Item
  │     ├─ Processing Element ─── Work-Item
  │     └─ ...
  │
  ├─ Compute Unit ─── Work-Group
  └─ ...
```

**关键特性**

1.SIMT

```cpp
// 所有工作项在同一计算单元内同步执行相同指令
__kernel void example(__global float* data) {
    int id = get_global_id(0);
    data[id] = data[id] * 2.0f;  // 所有工作项同时执行这个乘法
}
```

2.内存层次结构

```text
工作项 → 私有内存 (Private Memory)
    ↓
工作组 → 本地内存 (Local Memory) 
    ↓
所有工作组 → 全局内存 (Global Memory)
    ↓
主机 → 主机内存 (Host Memory)
```

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20251105230823.png)

```
快速访问 ←──────────────────────────→ 慢速访问
私有内存 → 本地内存 → 全局/常量内存缓存 → 全局内存
```

3.架构映射

- NVIDIA GPU
```text
OpenCL 概念      NVIDIA GPU 对应
==========      ================
Device          GPU 芯片 (如 GA102)
Compute Unit    流多处理器 (SM)
Processing Element CUDA 核心
Work-Group      线程块 (Thread Block)
Work-Item       线程 (Thread)
```

- AMD GPU
```text
OpenCL 概念      AMD GPU 对应
==========      =============
Device          GPU 芯片
Compute Unit    计算单元 (CU)
Processing Element 流处理器
```

**OpenCL 主机端代码**

示例：

```cpp
#include <CL/cl.h>
#include <stdio.h>

int main() {
    // 1. 设置平台和设备
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // 2. 创建上下文和命令队列
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
    
    // 3. 分配设备内存
    size_t size = 1024 * sizeof(float);
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    
    // 4. 编译内核
    const char* kernel_src = "..."; // 内核代码
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);
    
    // 5. 准备数据并传输到设备
    float host_input[1024], host_output[1024];
    // 初始化 host_input...
    clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, size, host_input, 0, NULL, NULL);
    
    // 6. 设置内核参数并启动
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
    
    size_t global_size = 1024, local_size = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    
    // 7. 读取结果
    clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, size, host_output, 0, NULL, NULL);
    
    // 8. 清理资源
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
```



