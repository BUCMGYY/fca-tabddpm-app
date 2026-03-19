# FCA-TabDDPM 表格数据增强平台

基于特征交叉注意力扩散模型的表格数据增强工具，面向存在类别不平衡问题的表格数据研究者开放使用。

## 功能

- **数据上传与配置**：上传CSV数据，自动识别特征类型，选择目标变量
- **模型训练与生成**：支持预训练权重微调/从头训练，提供四种增补策略
- **生成质量评估**：JSD / WD / PCD / 共现MAE 全维度评估

## 在线使用

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://你的用户名-fca-tabddpm.streamlit.app)

## 本地部署

```bash
git clone https://github.com/你的用户名/fca-tabddpm-app.git
cd fca-tabddpm-app
pip install -r requirements.txt
streamlit run app.py
```

## 引用

如果本工具对您的研究有帮助，请引用：

```
待补充
```

## 许可证

MIT License
