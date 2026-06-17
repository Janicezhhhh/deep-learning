# Anomaly Detection Project

这是一个简单的异常检测项目，使用隔离森林算法检测异常样本。

运行方式：

```bash
cd anomaly_detection_project
pip install -r requirements.txt
python train.py
```

项目内容：
- 生成正常数据和异常数据
- 使用 PyTorch 训练一个异常检测自编码器
- 输出异常分数并将结果保存到 `anomaly_results.txt`

示例输入输出：
- 正常点：`[[0.5, 0.3], [0.2, -0.1]]` → 异常分数接近 0
- 异常点：`[[10.0, 10.0], [-15.0, -15.0]]` → 异常分数接近 1
