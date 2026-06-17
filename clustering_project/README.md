# Clustering Project

这是一个简单的聚类项目，使用 PyTorch 实现一个基本的 k-means 算法。

运行方式：

```bash
cd clustering_project
pip install -r requirements.txt
python train.py
```

项目内容：
- 生成三个高斯分布聚类数据
- 使用 k-means 算法寻找聚类中心
- 输出每个点所属类别并保存聚类中心到 `cluster_centers.pth`

示例输入输出：
- 输入点：`[[1.0, 1.0], [5.0, 5.0], [-4.0, 0.0]]`
- 输出标签：`[0, 1, 2]`（或类似的聚类编号）
