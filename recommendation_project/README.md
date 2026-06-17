# Recommendation Project

这是一个简单的推荐系统项目，使用矩阵分解（MF）进行协同过滤推荐。

运行方式：

```bash
cd recommendation_project
pip install -r requirements.txt
python train.py
```

项目内容：
- 生成用户-物品交互矩阵
- 使用 PyTorch 训练矩阵分解模型
- 计算推荐精度并保存模型参数到 `recommendation_model.pth`

示例输入输出：
- 用户-物品交互：`[[1, 0, 1], [0, 1, 0]]`
- 预测评分：`[[0.9, 0.2, 0.85], [0.3, 0.95, 0.1]]`
