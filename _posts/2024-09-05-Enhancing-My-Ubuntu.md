---
layout: post
title: Enhancing My Ubuntu
subtitle: Ubuntu efficiency tools
date: 2024-09-06
author: Jinz
header-img: img/post-bg-swift2.jpg
catalog: true
tags:
  - Ubuntu
  - Terminal
  - Tmux
  - VPN
  - Zsh
---

# Ubuntu系统 环境配置

**包括搭建VPN，安装termianl，tumx，zsh，docker等工具**

*下面基本上是我个人的装机必配，可以做一个参考，所有操作在 **Ubuntu24** 版本上都验证过*

# VPN搭建 
## Clash GUI
- 确认计算机架构
  
  终端输入 `uname -a` 查看系统架构  
    
  ```
  Linux jinz-OptiPlex-7090 6.8.0-41-generic #41-Ubuntu SMP PREEMPT_DYNAMIC Fri Aug  2 20:41:06 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
  ```

  可以看出本机系统是 **x86** 的架构，可以选择 **amd64** 

- 下载 [clash_verge](https://github.com/Molunerfinn/PicGo/releases)
  
  > ubuntu24 下载会存在问题，需要安装[两个库](https://www.clashverge.dev/faq/linux.html)

- 配置 Clash
  
  打开下载的clash-verge,点击订阅，新建，将remote改为local，选择配置文件  

  ![](https://raw.githubusercontent.com/Jinz-HU/picRep/main/img/Screenshot%20from%202024-09-05%2011-00-54.png)
  
  **导入配置文件之后，一定要记住右键使用配置文件**  

  > 这里我的配置链接clash-verge找不到，所以选择这种方法。
  理论上可以直接使用配置链接导入

## Chrome

  ```dotnetcli
  sudo apt update -y
  sudo apt install wget -y
  wget "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" -O google.deb
  sudo dpkg -i google.deb
  ```
  
# VScode

> 不要在软件里面直接安装

```dotnetcli
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
```

```dotnetcli
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
```

```dotnetcli
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
```

```dotnetcli
sudo apt install code
```

依次运行上述命令即可

更新
```dotnetcli
sudo apt update
sudo apt upgrade
```

**安装插件配置 VScode**

- **C/C++ \  C/C++ Extension Pack \  C/C++ Themes**
- Chinese (Simplified)
- CMake \ CMake Tools
- **CodeSnap**
- GBK to UTF8 for vscode
- **Markdown All in One \  Markdown Preview Enhanced \ Markdown Preview Mermaid Support**
- MATLAB
- **Path Intellisense**
- **Python \  Pylance \ Python Debugger**
- **Vim**
seting.json 和 keybindings.json
[配置文件导入](https://github.com/Jinz-HU/Vim-VsCode)

# Zsh

```dotnetcli
sudo apt update
sudo apt install zsh
```

切换SHELL, **切记不要加 sudo**
```dotnetcli
chsh -s /bin/zsh
```


**Oh my zsh**

```dotnetcli
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```   

Oh-my-zsh 插件

- zsh-syntax-highlighting
```dotnetcli
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

- zsh-autosuggestions
```dotnetcli
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

- z
- web-search
- extract
> oh-my-zsh 自带

**完成上述步骤后，需要修改 ~/.zshrc 中的插件配置，并运行**
```dotnetcli
plugins=(git z web-search extract zsh-syntax-highlighting zsh-autosuggestions)
```

`source ~/.zshrc` source之后就配置成功了


# Tmxu 与 Terminator

**安装**

```dotnetcli
sudo apt install terminator
```

```dotnetcli
sudo apt-get install tmux
```

**配置**

配置文件：[Oh-my-tmux](https://github.com/gpakosz/.tmux?tab=readme-ov-file)

```dotnetcli
This configuration uses the following bindings:

<prefix> e opens the .local customization file copy with the editor defined by the $EDITOR environment variable (defaults to vim when empty)

<prefix> r reloads the configuration

C-l clears both the screen and the tmux history

<prefix> C-c creates a new session

<prefix> C-f lets you switch to another session by name

<prefix> C-h and <prefix> C-l let you navigate windows (default <prefix> n and <prefix> p are unbound)

<prefix> Tab brings you to the last active window

<prefix> - splits the current pane vertically

<prefix> _ splits the current pane horizontally

<prefix> h, <prefix> j, <prefix> k and <prefix> l let you navigate panes ala Vim

<prefix> H, <prefix> J, <prefix> K, <prefix> L let you resize panes

<prefix> < and <prefix> > let you swap panes

<prefix> + maximizes the current pane to a new window

<prefix> m toggles mouse mode on or off

<prefix> U launches Urlscan (preferred) or Urlview, if available

<prefix> F launches Facebook PathPicker, if available

<prefix> Enter enters copy-mode

<prefix> b lists the paste-buffers

<prefix> p pastes from the top paste-buffer

<prefix> P lets you choose the paste-buffer to paste from
```


[Tmux简介与基本操作](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

[Tmux简介](https://lrl52.top/794/tmux-configure/)（比阮一峰更详细）

# Vim

安装

```dotnetcli
sudo apt install vim
```

```dotnetcli
vim --version
```
[Ref](https://infotechys.com/install-vim-on-ubuntu-24-04/)

配置

> 暂时用vscode 不配置vim

# Docker

> 一定会用到

[安装](https://www.sysgeek.cn/install-docker-ubuntu/#1-%E7%AC%AC-1-%E6%AD%A5%EF%BC%9A%E6%9B%B4%E6%96%B0%E8%BD%AF%E4%BB%B6%E5%8C%85%E5%B9%B6%E5%AE%89%E8%A3%85%E5%BF%85%E8%A6%81%E8%BD%AF%E4%BB%B6)












