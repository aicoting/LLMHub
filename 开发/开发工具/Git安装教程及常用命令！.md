Git 是一款分布式版本控制系统，由 Linux 之父 Linus Torvalds 于 2005 年开发。它广泛用于代码版本管理，尤其在多人协作开发中发挥着核心作用。Git 能够高效地追踪文件变化，管理代码历史，支持分支、合并、回滚等多种操作，让开发者能够专注于代码本身，而不必担心版本混乱或数据丢失。

相比传统的集中式版本控制系统（如 SVN），Git 的分布式特性使得每个开发者本地都有完整的代码仓库和历史记录，即使在离线状态也能完成大部分操作。Git 同时也是 GitHub、GitLab、Gitee 等平台的底层技术，是现代软件开发的“标配工具”。

本篇文章将介绍Git的安装方法并整理 Git 中最常用的命令，帮助你快速上手、查阅与解决日常开发中的版本控制问题。

<font style="color:rgb(25, 27, 31);">关于深度学习和大模型相关的知识和前沿技术更新，请关注公众号</font><font style="color:rgb(25, 27, 31);background-color:rgb(246, 246, 246);">算法coting</font><font style="color:rgb(25, 27, 31);">！</font>

<font style="color:rgb(25, 27, 31);">所有相关源码示例、流程图、模型配置与知识库构建技巧，我也将持续更新在Github：</font>[**<font style="color:rgb(25, 27, 31);">LLMHub</font>**](https://github.com/zhangting-hit/LLMHub)<font style="color:rgb(25, 27, 31);">，欢迎关注收藏！</font>

## 安装教程
首先到官网[https://git-scm.com/downloads/win](https://git-scm.com/downloads/win)下载git安装包。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754112854147-bd346125-f7aa-4327-ae82-00737ebf665c.png)

下载完之后进行双击安装。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754112970428-42f6d0b3-495b-417f-a706-1b0a2d3712b7.png)

这里最好安装到非系统盘。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754112978765-025dbe97-9584-45be-a6f6-67088cf2b8f9.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754112989263-f3953ef5-6ebd-4ba7-8183-4fdc3b40c80d.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754113073533-f845fd2d-4f37-493a-b8e7-ec838e50dbc5.png)

这里要选择git的编辑器，官方默认的是vim，如果不会vim的可以选择vscode或者notpad++等等。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754113155469-f677a364-f8a4-4ebf-a4e9-3619ec177aa5.png)

接下来一路next直至安装完成。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754113513083-221cb601-32f2-4d89-96d0-9b156780665a.png)



安装完成后就可以使用git命令行啦。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754113718761-84dfd320-416c-4072-bd95-44ba161b6f02.png)

在命令行提交代码的时候要首先登录自己的github账号，才可以push成功。

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754120534422-02facaed-7062-40da-affc-2b11da0e4841.png)

![](https://cdn.nlark.com/yuque/0/2025/png/28454971/1754120563214-fc0fda38-2cbd-4fc4-97fd-b01969ef1c8b.png)

## 常用命令
### 2.1 基本配置
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.editor "code"       # 设置 VS Code 为默认编辑器
git config --global -l                       # 查看当前配置
```

---

### 2.2 仓库初始化与克隆
```bash
git init                      # 初始化本地 Git 仓库
git clone <repo-url>         # 克隆远程仓库
git clone <url> <folder>     # 克隆到指定文件夹
```

---

### 2.3 文件操作
```bash
git status                   # 查看当前状态
git add <file>               # 添加文件到暂存区
git add .                    # 添加当前目录下所有更改文件
git rm <file>                # 删除文件并提交
git mv <old> <new>           # 重命名文件
```

---

### 2.4 提交更改
```bash
git commit -m "commit message"     # 提交暂存区到本地仓库
git commit -am "msg"               # 跳过 add，直接提交已跟踪文件的修改
```

---

### 2.5 查看历史与差异
```bash
git log                          # 查看提交历史
git log --oneline --graph        # 简洁图形化显示提交历史
git diff                         # 查看当前工作区与暂存区差异
git diff --staged                # 查看暂存区与上次提交差异
```

---

### 2.6 分支管理
```bash
git branch                       # 列出本地分支
git branch <branch-name>         # 创建新分支
git checkout <branch-name>       # 切换分支
git checkout -b <branch-name>    # 创建并切换新分支
git merge <branch-name>          # 合并指定分支到当前分支
git branch -d <branch-name>      # 删除分支
```

---

### 2.7 远程仓库操作
```bash
git remote -v                        # 查看远程仓库地址
git remote add origin <url>         # 添加远程仓库
git push -u origin main             # 第一次推送并设置 upstream 分支
git push                            # 推送更改
git pull                            # 拉取并合并远程更改
```

---

### 2.8 撤销与还原
```bash
git restore <file>                 # 撤销工作区修改（Git 2.23+）
git reset HEAD <file>             # 撤销暂存区的某个文件
git reset --hard                  # 恢复到最后一次提交（⚠️危险）
git clean -fd                     # 删除未被 Git 跟踪的文件和文件夹
```

---

### 2.9 标签管理
```bash
git tag                            # 列出所有标签
git tag v1.0                       # 创建标签
git tag -d v1.0                    # 删除标签
git push origin v1.0              # 推送标签
```

---

### 2.10 临时工作区（Stash）
```bash
git stash                          # 暂存当前修改
git stash list                     # 查看所有暂存
git stash pop                      # 恢复最近暂存并删除
git stash apply stash@{0}          # 应用指定暂存但保留记录
```

---

## 小技巧
+ `git log -p -n 1`：查看最近一次提交内容
+ `git commit --amend`：修改上一次提交信息
+ `git rebase -i HEAD~3`：交互式变基，修改最近 3 次提交
+ `git cherry-pick <commit>`：把某次提交复制到当前分支

---

> 💡 温馨提示：**Git 命令强大但也有风险，尤其是 **`**reset --hard**`** 和 **`**rebase**`**，建议先做好备份或在测试分支中操作！**
>





