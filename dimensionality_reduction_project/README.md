# Dimensionality Reduction Project

这是一个简单的降维项目，使用 PyTorch 实现一个小型自编码器。

运行方式：

```bash
cd dimensionality_reduction_project
pip install -r requirements.txt
python train.py
```

项目内容：
- 生成 4 维合成数据
- 使用自编码器将数据降维到 2 维表示
- 输出示例输入和编码后的低维向量

示例输入输出：
- 输入向量：`[2.0, -1.0, 0.5, 3.0]`
- 输出编码：`[0.12, -0.04]`（示例值，训练后会生成低维表示）
