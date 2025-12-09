# GitHub 上传指南

## 📋 上传前检查

### 已排除的大文件

✅ `.gitignore` 文件已配置，以下文件**不会**上传到GitHub：

- ✅ **数据集文件**
  - `data/raw/` - 所有原始CSV文件
  - `data/processed/` - 所有处理后的.npz文件

- ✅ **模型检查点**
  - `checkpoints/` - 所有训练好的模型
  - 所有 `.pt`, `.pth`, `.ckpt` 文件

- ✅ **日志文件**
  - `logs/` - 训练日志
  - `runs/`, `wandb/` - TensorBoard日志

- ✅ **输出文件**
  - `outputs/`, `results/` - 实验结果
  - 所有图片、视频文件

---

## 🚀 上传步骤

### 步骤1: 初始化Git仓库

```bash
# 进入项目目录
cd D:\DRIFT\A\traffic-world-model

# 初始化Git仓库（如果还没初始化）
git init

# 查看当前状态
git status
```

---

### 步骤2: 检查将要上传的文件

```bash
# 查看哪些文件会被上传（绿色的会上传）
git status

# 查看哪些文件被忽略
git status --ignored
```

**重要：** 确保 `data/raw/`, `checkpoints/`, `logs/` 等大文件夹显示为被忽略状态

---

### 步骤3: 添加文件到Git

```bash
# 添加所有文件（.gitignore会自动排除大文件）
git add .

# 再次检查
git status
```

---

### 步骤4: 创建第一次提交

```bash
# 提交到本地仓库
git commit -m "Initial commit: Traffic World Model project"
```

---

### 步骤5: 在GitHub创建远程仓库

1. 打开 GitHub: https://github.com
2. 点击右上角的 `+` → `New repository`
3. 填写信息：
   - **Repository name**: `traffic-world-model`
   - **Description**: `Multi-agent latent world model for drone-based vehicle trajectory prediction`
   - **Public** 或 **Private**（根据需要选择）
   - ⚠️ **不要**勾选 `Initialize this repository with a README`（我们已经有了）
4. 点击 `Create repository`

---

### 步骤6: 连接到远程仓库

复制GitHub显示的命令，或使用以下命令：

```bash
# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/traffic-world-model.git

# 验证远程仓库
git remote -v
```

---

### 步骤7: 推送到GitHub

```bash
# 推送到远程仓库
git push -u origin main

# 如果提示分支名是master而不是main，使用：
git push -u origin master

# 或者先重命名分支为main
git branch -M main
git push -u origin main
```

---

### 步骤8: 验证上传

1. 刷新GitHub页面
2. 检查文件是否上传成功
3. **确认以下文件夹不存在**：
   - ❌ `data/raw/`
   - ❌ `data/processed/` 中的.npz文件
   - ❌ `checkpoints/`
   - ❌ `logs/`

---

## 📝 后续更新

### 日常提交流程

```bash
# 1. 查看修改
git status

# 2. 添加修改的文件
git add .

# 3. 提交
git commit -m "描述你的修改"

# 4. 推送到GitHub
git push
```

### 提交信息示例

```bash
git commit -m "Add multi-site data preprocessing"
git commit -m "Update training hyperparameters"
git commit -m "Fix bug in encoder forward pass"
git commit -m "Add attention visualization"
```

---

## ⚠️ 常见问题

### 问题1: 文件太大无法上传

**错误信息：**
```
remote: error: File data/raw/A/drone_1.csv is 123.45 MB; this exceeds GitHub's file size limit of 100 MB
```

**解决方案：**

1. **检查.gitignore是否生效**
   ```bash
   git rm --cached -r data/raw/
   git commit -m "Remove large files"
   git push
   ```

2. **如果已经提交了大文件，需要从历史中删除**
   ```bash
   # 使用git filter-branch（慎重！）
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch -r data/raw/" \
     --prune-empty --tag-name-filter cat -- --all

   # 强制推送
   git push origin --force --all
   ```

---

### 问题2: .gitignore不生效

**原因：** 文件已经被Git追踪

**解决方案：**
```bash
# 从Git缓存中移除
git rm -r --cached data/raw/
git rm -r --cached checkpoints/
git rm -r --cached logs/

# 重新添加（这次会应用.gitignore）
git add .
git commit -m "Apply .gitignore rules"
git push
```

---

### 问题3: 推送被拒绝

**错误信息：**
```
! [rejected] main -> main (fetch first)
```

**解决方案：**
```bash
# 先拉取远程更改
git pull origin main --rebase

# 再推送
git push origin main
```

---

### 问题4: 验证.gitignore配置

```bash
# 测试某个文件是否会被忽略
git check-ignore -v data/raw/A/drone_1.csv

# 应该输出类似：
# .gitignore:44:data/raw/    data/raw/A/drone_1.csv
```

---

## 🔍 检查仓库大小

### 在推送前检查

```bash
# 查看将要上传的文件大小
git ls-files | xargs du -ch

# 查看整个仓库大小
du -sh .git/
```

### 推荐的仓库大小

- ✅ **< 100 MB**: 理想
- ⚠️ **100 MB - 1 GB**: 可接受
- ❌ **> 1 GB**: 过大，需要清理

---

## 📦 如果需要分享数据

如果需要分享数据集或模型，使用以下方式：

### 方案1: GitHub Releases

1. 在GitHub仓库页面，点击 `Releases`
2. 点击 `Create a new release`
3. 上传 `.zip` 文件（限制2GB）
4. 在README中添加下载链接

### 方案2: Git LFS (大文件存储)

```bash
# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "*.npz"
git lfs track "*.pt"

# 添加.gitattributes
git add .gitattributes

# 正常提交和推送
git add data/processed/train_episodes.npz
git commit -m "Add training data"
git push
```

### 方案3: 外部存储

推荐使用：
- Google Drive
- Dropbox
- OneDrive
- 百度网盘
- 阿里云OSS

在README中添加下载链接

---

## 📄 README建议

在GitHub仓库主页添加以下说明：

```markdown
## Data

Due to file size limitations, the dataset is not included in this repository.

### Download Data

- **Raw data**: [Download link] (XX GB)
- **Processed data**: [Download link] (XX GB)
- **Pre-trained models**: [Download link] (XX MB)

### Data Structure

After downloading, extract to:
```
traffic-world-model/
├── data/
│   ├── raw/
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── processed/
│       ├── train_episodes.npz
│       ├── val_episodes.npz
│       └── test_episodes.npz
```
```

---

## ✅ 最终检查清单

上传前确认：

- [ ] `.gitignore` 文件存在
- [ ] 运行 `git status` 检查没有大文件
- [ ] `data/raw/` 不在待提交列表
- [ ] `checkpoints/` 不在待提交列表
- [ ] `logs/` 不在待提交列表
- [ ] README.md 已更新
- [ ] 代码可以正常运行
- [ ] 敏感信息已删除（API keys, 密码等）

---

## 🎯 快速命令速查

```bash
# 完整上传流程
cd D:\DRIFT\A\traffic-world-model
git init
git add .
git commit -m "Initial commit: Traffic World Model"
git remote add origin https://github.com/YOUR_USERNAME/traffic-world-model.git
git branch -M main
git push -u origin main

# 日常更新
git add .
git commit -m "Your commit message"
git push

# 删除已追踪的大文件
git rm -r --cached data/raw/
git commit -m "Remove large files"
git push

# 检查.gitignore是否生效
git check-ignore -v data/raw/A/drone_1.csv
git status --ignored
```

---

**最后更新:** 2025-12-09
