---
layout: post
title: Resolve the Github connection issue
subtitle: Resolve the timeout issue
date: 2024-08-14
author: Jinz
header-img: img/post-bg3.jpg
catalog: true
tags:
  - Github
---

# 解决远程访问 Github 超时问题

> 科学上网访问 Github 时拉取和推送远程仓库容易出现超时问题

## 确认能访问 Github

打开终端输入 `ssh -T git@github.com` ，如果返回 ` You've successfully authenticated` 证明能够成功访问。

## 解决方法

### 绕过国内 DNS 解析

> 参考这篇[博客](https://jasonkayzk.github.io/2019/10/10/%E5%85%B3%E4%BA%8E%E4%BD%BF%E7%94%A8Git%E6%97%B6push-pull%E8%B6%85%E6%97%B6-%E4%BB%A5%E5%8F%8AGithub%E8%AE%BF%E9%97%AE%E6%85%A2%E7%9A%84%E8%A7%A3%E5%86%B3%E5%8A%9E%E6%B3%95/)

进入 https://www.ipaddress.com/

查找以下三个链接的 DNS 解析地址

```
github.com
assets-cdn.github.com
github.global.ssl.fastly.net
```

以 github.com 为例

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408141449990.jpg)

查找到对应 IP 地址后，在 Windows 下进入 `C:\Windows\System32\drivers\etc` , 把 `140.82.112.3    github.com` 加入到 hosts 后面，然后输入 `ipconfig /flushdns` 刷新 DNS 缓存

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408141457839.jpg)

> 如果有多个 DNS 解析地址的话就都加上

**试过这种方法但是不能解决我的问题，推送还是超时**

### 设置代理端口

首先，取消代理

```html
git config --global --unset http.proxy 
git config --global --unset https.proxy
```

刷新一下 DNS 缓存 `ipconfig /flushdns`

查看本机代理端口

![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/202408141509210.png)

设置代理

```html
git config --global http.proxy http://127.0.0.1:7890 
git config --global https.proxy http://127.0.0.1:7890
```

输入 `git config --global -l` 检查是否配置成功

**OK，现在速度就飞快了**
