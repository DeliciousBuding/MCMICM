# 美赛环境配置指南 (Team 2617892)

这份文档旨在帮助快速配置 VS Code、LaTeX 和 Git 环境。

## 1. 安装编辑器与基础环境
- **VS Code**: [官网下载](https://code.visualstudio.com/)
- **Git**: [官网下载](https://git-scm.com/downloads) (安装后才能使用 GitHub 同步)
- **LaTeX (精简版)**: 推荐 [MiKTeX](https://miktex.org/download) (仅 200MB+，缺包自动下载)

## 2. 安装 VS Code 插件
打开 VS Code，在左侧扩展商店 (Ctrl+Shift+X) 搜索并安装：
1. `Chinese (Simplified)` - 简体中文包
2. `LaTeX Workshop` - LaTeX 支持

## 3. 注册并连接 GitHub (每个人用自己的账号)
1. **注册**: 去 [GitHub 官网](https://github.com/) 每个人注册一个属于自己的账号。
2. **报名字**: 把你的用户名发给队长，让队长点击 `Settings -> Collaborators` 邀请你加入仓库（**记得去邮箱点确认邀请**）。
3. **配置身份 (非常重要)**:
   在 VS Code 里按 `Ctrl + ~` 打开终端，依次输入以下两行命令（把名字和邮箱换成你自己的）：
   - `git config --global user.name "你的中文名或英文名"`
   - `git config --global user.email "你的GitHub邮箱地址"`
   *这样提交记录里就会显示这行代码是你写的！*
4. **登录 VS Code**: 点击左下角账户图标，使用**你自己的账号**登录。
5. **拉取代码**: 按 `Ctrl + Shift + P` -> 输入 `Git: Clone` -> `Clone from GitHub` -> 选择 `MCMICM`。

## 4. 日常操作 (提交与同步)
- **修改文件**后，点击左侧 `源代码管理` 图标 (Ctrl+Shift+G)。
- 在 `消息` 框输入这次改了什么 (例如: "更新了建模部分的公式")。
- 点击 **提交 (Commit)**，然后点击 **同步更改 (Sync / Push)** 即可。

---
祝竞赛顺利！🚀
