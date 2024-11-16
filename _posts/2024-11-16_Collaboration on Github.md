---
layout: post
title: Collaboration on Github
subtitle: SOP for programming using Git [IGA Group]
date: 2024-11-16
author: Jinz
header-img: img/post-bg4.jpg
catalog: true
tags:
  - Git
  - Github
  - Terminal
---

# 使用Github进行协作编程

# 准备工作
1. [配置Git](https://www.icode504.com/posts/69.html)
2. 在Github，[创建仓库](https://blog.csdn.net/u012693479/article/details/114892130)
3. [加入协作人员](https://docs.github.com/zh/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository)

经过上述操作就可以开始共同开发代码了，在这之前需要熟悉一些Git的[基本操作命令](https://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html)和[流程](https://www.bilibili.com/video/BV1yo4y1d7UK/?spm_id_from=333.337.search-card.all.click&vd_source=062b5b4a482e075edce4af60daf92005)。

# Git开发流程
## 创建项目
* 团队成员在[IGA](https://github.com/DLUT-HPIGA)(团队组织)创建仓库
* 开发者fork仓库到个人账号
* 开发者clone仓库到本地进行开发

## 开发规范
```
clone到本地的仓库默认分支在主分支
建立新分支，新分支以本次操作命名，如：add_chat_function
当开发期间团队仓库分支发生更改，并可能与正在开发的任务发生冲突时，切换到main分支，pull新的内容，然后切换回开发分支，使用rebase命令进行处理，或者fetch新的代码
```

### Fork项目

进入目标仓库，fork项目到自己的仓库

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116104701.png)

然后你可以在自己仓库中找到fork的仓库

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116104809.png)

### 创建分支

1. 将仓库clone到本地

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116105104.png)

2. 切换分支，以**本次操作为名**

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116105339.png)

### 功能开发

1. 在新分支下进行代码开发

2. 开发完成后**提交 commit**

<font color="red">每一次代码能跑之后commit一次！！！</font>

3. push代码到你的仓库

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116115636.png) 

### Pull Request

1. 切换分支，提Pull Request

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116122435.png)

2. Pull requests

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/20241116122114.png)

3. 管理员merge pull request

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/1031a3add6e2ad939feb603abbd6a8d.png)

进入Github主页面点进去就能看见

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/744228b2911262838017cfa43488708.png)

> 这里也可以按着readme里面写的规范进行开发，可以省略上面的fork步骤

## commit规范

[参考](https://juejin.cn/post/6844903606815064077)

```
<type>(<IGA>): <message>
```

**eg**: `feat(fehead_build): add document`

``` MAKEFILE
feat: 新特性
fix: 修改问题
refactor: 代码重构
docs: 文档修改
style: 代码格式修改, 注意不是 css 修改
test: 测试用例修改
chore: 其他修改, 比如构建流程, 依赖管理
```