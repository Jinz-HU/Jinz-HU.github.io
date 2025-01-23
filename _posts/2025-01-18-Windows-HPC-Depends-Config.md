---
layout: post
title: 2025-01-18-HPC-Depends-Config
subtitle: Install the depends lib (Eigen/MKL/PETSc) in windows
date: 2024-07-23
author: Jinz
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
  - Visual Studio
  - Windows
  - HPC
---

# 安装高性能计算库
- Eigen
- MKL
- PETSc
> 记录windows配置过程

# Eigen
## 1. 安装[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
      下载 .zip 文件直接安装即可
## 2. Visual Studio 配置
  - 右击peoject，进入项目属性，选择 **所有配置**，在包含目录中加入Eigen路径，即可
  ![属性](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20250118154831.png)
  - 直接将Eigen所有文件移入工程中，创建一个文件夹放这些需要的库，再在属性中写入**相对路径**

## 3. 测试代码
``` c++
#include <iostream>
#include <Eigen\Dense>

using namespace std;

typedef Eigen::Matrix<int, 3, 3> Matrix3i;

int main()
{
    /*
    Matrix的初始化方法
    Eigen::Matrix<int, 3, 3>
    int 代表Matrix的数据类型，3，3 分别代表 rows， cols
    Matrix3i m1;
    m1(0,0) = 1
    m1(0,1) = 2
    m1(0,2) = 3
    ...
    或者用 m1 << 1,2,3 ...
    */

    Matrix3i m1;
    m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    cout << "m1 = \n" << m1 << endl;

    Matrix3i m2;
    m2 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    cout << "m2 = \n" << m2 << endl;

    cout << "m1 * m2 = \n" << (m1 * m2) << endl;

    return 0;
}
```

# MKL
## 1. 安装 [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
      建议直接安装到C盘下面

## 2. Visual Studio 配置
- 进入**配置属性—> VC++**目录输入下面部分
```
可执行文件目录：
C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\intel64
包含目录：
C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include
库目录：
C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib
C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib
```
- 进入 **链接器->输入->附加依赖项** 添加下面内容
```
mkl_intel_ilp64.lib
mkl_intel_thread.lib
mkl_core.lib
libiomp5md.lib
```
- 进入 **配置属性->Intel Libraries for oneAPI**
      Use oneMKL = Parallel

## 3. 测试代码
``` c++
#include<stdio.h>
#include<stdlib.h>

#include"mkl.h"
#include"mkl_lapacke.h"
#define n 4

void main() {

	int matrix_order = LAPACK_COL_MAJOR;
	char jobvl = 'N';
	char jobvr = 'V';
	double A[n * n] = {
		 0.35,  0.09, -0.44,  0.44,
		 0.09,  0.07, -0.33, 0.52,
		-0.44, -0.33, -0.03, -0.1,
		0.44,  0.52,  -0.1,  0.35 };//4*4矩阵
	int lda = n;
	double wr[n] = { 0 };
	double wi[n] = { 0 };
	double vl[n * n];
	int ldvl = n;
	double vr[n * n];
	int ldvr = n;
	int info = LAPACKE_dgeev(matrix_order, jobvl, jobvr, n, A, lda, wr, wi, vl, ldvl, vr, ldvr);
	//int info=0;

	if (info == 0) {
		int i = 0;
		int j = 0;
		int flag = 0;//区分复特征值的顺序
		for (i = 0; i < n; i++) {
			printf("eigenvalue %d:", i);
			printf("%.6g + %.6gi\t", wr[i], wi[i]);
			printf("\n");
			printf("right eigenvector: ");
			if (wi[i] == 0)
			{
				for (j = 0; j < ldvr; j++) {
					printf("%.6g\t", vr[i * n + j]);
				}
			}
			else if (flag == 0)//如果该复特征值为这对复特征值的第一个则
			{
				flag = 1;
				for (j = 0; j < ldvr; j++)
				{
					printf("%.6g + %.6gi\t", vr[i * n + j], vr[(i + 1) * n + j]);
				}
			}
			else if (flag == 1)//如果该复特征值为这对复特征值的第二个则
			{
				flag = 0;
				for (j = 0; j < ldvr; j++)
				{
					printf("%.6g + %.6gi\t", vr[(i - 1) * n + j], -vr[i * n + j]);
				}
			}
			printf("\n");
		}
		getchar();//必须要有这句
		printf("SUCCESS\n");
	}

	system("pause");
}
```


