# Sequence Modeling Project

这是一个简单的序列预测项目，使用 LSTM 预测时间序列的下一个值。

运行方式：

```bash
cd sequence_modeling_project
pip install -r requirements.txt
python train.py
```

项目内容：
- 生成合成时间序列数据
- 使用 PyTorch LSTM 进行序列预测
- 计算测试集均方误差并保存模型参数

示例输入输出：
- 输入序列：`[0.1, 0.2, 0.3, 0.4]` → 预测下一个值
- 输出预测：`0.5`（或类似的连续值）
