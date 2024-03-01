# Git 教程


 ```
# Git 简介

Git 是一款开源的分布式版本控制系统，用于追踪文件更改和协调多人之间的工作。它最初是由 Linus Torvalds 开发的，用于管理 Linux 内核的源代码。Git 具有高效、安全、可扩展等特点，已成为当今最受欢迎的版本控制系统。

Git 的主要特点有：

1. 快：Git 具有良好的性能，特别是在处理大型项目时表现出色。

2. 安全：Git 提供了数据完整性，通过哈希树（Merkle Tree）结构确保数据不易损坏。

3. 可扩展：Git 支持插件和扩展，可以根据需求定制功能。

4. 分支和合并：Git 支持分支操作，方便开发者在不同功能分支上进行开发，然后将分支合并到主分支。

5. 压缩存储：Git 采用数据压缩算法，减少存储空间占用。

6. 远程协作：Git 支持远程仓库，方便团队成员之间协作开发。

## Git 安装与配置

### 安装 Git

在我国，推荐使用 Git 社区版（Git-cmd）进行安装。请按照以下步骤进行安装：

1. 打开终端。

2. 输入以下命令下载 Git：

```
wget -c https://github.com/git-scm/git/releases/download/v2.34.1/git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
```

3. 解压下载的文件：

```
tar -zxvf git-2.34.1-64-bit-posix-seh-rt_64.tar.gz
```

4. 进入解压后的目录：

```
cd git-2.34.1
```

5. 编译并安装 Git：

```
./configure
make
sudo make install
```

6. 安装完成后，检查 Git 是否安装成功：

```
git --version
```

如果显示 Git 版本信息，说明安装成功。

### 配置 Git

1. 设置 Git 用户名和邮箱：

```
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

2. 设置 Git 参数界面：

```
git config --global gitprompt.show '%s'
```

3. 设置 Git 分支参数：

```
git config --global branch.autosync true
git config --global branch.default master
```

4. 设置 Git 远程仓库：

```
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

5. 创建 .gitignore 文件，忽略编译产物和日志文件：

```
touch .gitignore
echo "# 编译产物" >> .gitignore
echo "# 日志文件" >> .gitignore
```

6. 创建 .gitconfig 文件，配置 Git 参数：

```
touch .gitconfig
echo "[core]" >> .gitconfig
echo "autocrlf = true" >> .gitconfig
echo "filemode = true" >> .gitconfig
```

至此，Git 安装与配置完成。接下来，您可以开始使用 Git 进行版本控制。


 ```
# Git 教程

## Git 基本操作

### 1.1 初始化仓库

```bash
git init
```

### 1.2 添加文件

```bash
git add 文件名
```

### 1.3 提交代码

```bash
git commit -m "提交信息"
```

### 1.4 创建分支

```bash
git checkout -b 分支名
```

### 1.5 切换分支

```bash
git checkout 分支名
```

### 1.6 查看分支

```bash
git branch
```

### 1.7 合并分支

```bash
git merge 分支名
```

### 1.8 删除分支

```bash
git branch -d 分支名
```

## Git 分支管理

### 2.1 分支策略

- 主分支：master/main
- 开发分支：develop
- 功能分支：feature
- 发布分支：release
-  hotfix 分支：hotfix

### 2.2 分支命名规范

- 以字母开头，可以包含数字和字母
- 推荐使用短命名，便于识别
- 避免使用中文和特殊字符

### 2.3 分支管理流程

1. 从主分支创建开发分支
2. 开发过程中，不断提交代码，合并到开发分支
3. 开发分支达到一定稳定性，创建发布分支
4. 发布分支进行测试，修复 bug，合并到发布分支
5. 发布分支稳定后，合并到主分支
6. 主分支合并完成后，删除开发分支和发布分支

## Git 代码提交与拉取

### 3.1 代码提交

- 提交时添加描述性的提交信息，便于后人查看
- 遵循 [Conventional Changelog](https://conventionalchangelog.com/) 规范

### 3.2 代码拉取

```bash
git pull
```

## Git 冲突解决

### 4.1 冲突检测

- 相同文件名，内容不同：冲突
- 相同文件名，内容相同：未冲突
- 不同文件名：未冲突

### 4.2 冲突解决

1. 手动解决：比较冲突文件，修改后合并
2. 使用 [`git-median`](https://github.com/ginatls/git-median) 工具自动解决冲突

## Git 标签管理

### 5.1 创建标签

```bash
git tag -a 标签名 -m "描述"
```

### 5.2 推送标签

```bash
git push origin 标签名
```

### 5.3 拉取标签

```bash
git pull origin 标签名
```

### 5.4 删除标签

```bash
git tag -d 标签名
```

## Git 高级功能与应用

### 6.1 远程仓库管理

1. 添加远程仓库

```bash
git remote add 仓库名 远程仓库地址
```

2. 删除远程仓库

```bash
git remote remove 仓库名
```

3. 推送/拉取仓库

```bash
git push
git pull
```

### 6.2 代码审查

1. 创建审查分支

```bash
git checkout -b 审查分支名
```

2. 提交代码审查请求

```bash
git review
```

3. 审查代码

- 查看代码：`git log 审查分支名`
- 提交反馈：`git review <反馈编号>`

### 6.3 代码部署

1. 创建部署脚本

```bash
git deploy
```

2. 部署代码

```bash
./部署脚本
```

### 6.4 代码备份

1. 创建备份脚本

```bash
git backup
```

2. 备份代码

```bash