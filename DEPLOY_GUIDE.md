# FCA-TabDDPM Streamlit 部署完整指南

## 一、项目目录结构

部署前请确保你的项目目录如下：

```
fca-tabddpm-app/
├── app.py                        # 主程序（已提供）
├── requirements.txt              # 依赖列表（已提供）
├── README.md                     # 项目说明（已提供）
├── .gitignore                    # Git忽略规则（已提供）
├── .streamlit/
│   └── config.toml               # Streamlit主题配置（已提供）
└── pretrained/
    └── best_model.pt             # 你的预训练权重（需自行放入）
```

## 二、本地测试

在上传GitHub之前，先在本地确认能正常运行。

### 步骤1：创建虚拟环境

```bash
conda create -n fca-app python=3.10 -y
conda activate fca-app
```

### 步骤2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤3：放入预训练权重

```bash
mkdir -p pretrained
cp /你的路径/best_model.pt pretrained/
```

### 步骤4：本地运行

```bash
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501，逐一测试三个标签页的功能。

## 三、上传到GitHub

### 步骤1：创建GitHub仓库

1. 登录 https://github.com
2. 点击右上角 "+" → "New repository"
3. 仓库名填：`fca-tabddpm-app`
4. 选择 Public（Streamlit Cloud免费版要求公开仓库）
5. 不要勾选 "Add a README file"（我们已有）
6. 点击 "Create repository"

### 步骤2：初始化并推送

在你的项目目录下执行：

```bash
cd fca-tabddpm-app

git init
git add .
git commit -m "初始提交：FCA-TabDDPM数据增强平台"
git branch -M main
git remote add origin https://github.com/你的用户名/fca-tabddpm-app.git
git push -u origin main
```

### 注意：best_model.pt 文件大小

你的权重文件为727KB，远小于GitHub的100MB限制，可以直接提交。
如果未来权重文件超过100MB，需要使用Git LFS：

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add pretrained/best_model.pt
git commit -m "添加预训练权重（LFS）"
git push
```

## 四、部署到Streamlit Cloud

### 步骤1：登录Streamlit Cloud

1. 打开 https://share.streamlit.io
2. 点击 "Sign in with GitHub"，授权GitHub账号

### 步骤2：创建应用

1. 点击 "New app"
2. 填写以下信息：
   - **Repository**: `你的用户名/fca-tabddpm-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. 点击 "Advanced settings"（可选）：
   - Python version: 3.10
4. 点击 "Deploy!"

### 步骤3：等待部署

首次部署需要5-10分钟（安装PyTorch较慢），部署过程中可以查看日志。
部署成功后会获得一个公开链接，格式为：

```
https://你的用户名-fca-tabddpm-app.streamlit.app
```

## 五、部署常见问题

### Q1：部署时报错 "torch安装失败"

Streamlit Cloud 默认提供1GB内存，PyTorch完整版可能超限。
解决方案：将 requirements.txt 中的 torch 改为 CPU 版本：

```
torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Q2：部署后页面加载很慢

首次访问需要冷启动，约30秒。后续访问会缓存，加载更快。
可以在 .streamlit/config.toml 中添加：

```toml
[server]
enableStaticServing = true
```

### Q3：上传的CSV文件大小限制

默认最大200MB（已在config.toml中配置）。Streamlit Cloud免费版
总资源上限1GB，上传大文件可能导致内存不足。

### Q4：模型训练时间过长/超时

Streamlit Cloud 有运行时间限制。建议：
- 微调模式设置训练轮数不超过500
- 从头训练建议在本地完成后上传权重
- 生成样本数量建议不超过1000

### Q5：如何更新应用

修改代码后推送到GitHub即可自动重新部署：

```bash
git add .
git commit -m "更新描述"
git push
```

Streamlit Cloud 会自动检测GitHub的变更并重新部署。

## 六、自定义域名（可选）

Streamlit Cloud 支持自定义域名：

1. 在应用设置页面点击 "Custom subdomain"
2. 输入你想要的子域名，如 `fca-tabddpm`
3. 最终链接为 `https://fca-tabddpm.streamlit.app`

## 七、部署后检查清单

- [ ] 三个标签页均可正常切换
- [ ] 上传CSV后特征类型识别正确
- [ ] 预训练权重加载成功（控制台显示匹配参数数量）
- [ ] 模型训练过程有进度条和损失曲线
- [ ] 生成的合成数据可正常下载
- [ ] 质量评估指标正常计算和展示
- [ ] 共现概率矩阵热图正常渲染
