import csv
import importlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS = 2000
FEATURES = ("x1", "x2", "x3", "x4")
WINDOW = tuple(f"t{i}" for i in range(1, 11))
LOAD_LOCK = Lock()


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    default: int
    minimum: int
    maximum: int
    step: int = 1


ROWS = Parameter("samples", "样本行数", 120, 1, MAX_ROWS)
SEED = Parameter("seed", "随机种子", 42, 0, 2**31 - 1)


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    module: str
    weights: str
    class_name: str | None
    arguments: tuple = ()


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    method: str
    summary: str
    input_hint: str
    columns: tuple[str, ...]
    output: tuple[str, ...]
    dataset: tuple[str, ...]
    models: tuple[Model, ...]
    parameters: tuple[Parameter, ...]
    predict: Callable
    generate: Callable


def module_for(model):
    return importlib.import_module(f"{model.module}.train")


@lru_cache(maxsize=8)
def load(model):
    import torch

    weights = torch.load(ROOT / model.module / model.weights, map_location="cpu", weights_only=True)
    if model.class_name is None:
        return weights
    network = getattr(module_for(model), model.class_name)(*model.arguments)
    network.load_state_dict(weights)
    return network.eval()


def classification(torch, model, rows):
    probability = torch.sigmoid(model(rows)).flatten()
    return torch.column_stack((probability >= 0.5, probability)).tolist()


def regression(torch, model, rows):
    return model(rows).tolist()


def clustering(torch, centers, rows):
    distance, group = torch.cdist(rows, centers).min(dim=1)
    return torch.column_stack((group, distance)).tolist()


def reduction(torch, model, rows):
    encoded, decoded = model(rows)
    return torch.cat((encoded, decoded), dim=1).tolist()


def anomaly(torch, model, rows):
    _, decoded = model(rows)
    return (rows - decoded).square().mean(dim=1, keepdim=True).tolist()


def recommendation(torch, model, rows):
    return model(rows[:, 0].long(), rows[:, 1].long()).unsqueeze(1).tolist()


def labeled(module, options):
    x, y = module.make_data(options["samples"], options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), y.flatten().tolist())]


def cluster_data(module, options):
    x, y, _ = module.make_data(options["samples"], options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), y.tolist())]


def reduction_data(module, options):
    return module.make_data(options["samples"], options["seed"]).tolist()


def anomaly_data(module, options):
    x, labels = module.make_data(options["normal"], options["anomalies"], options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), labels.tolist())]


def sequence_data(module, options):
    x, y = module.make_sequence_data(options["samples"] + len(WINDOW) + 1, len(WINDOW), options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), y.flatten().tolist())]


def rating_data(module, options):
    matrix, mask = module.make_interaction_matrix(50, 30, options["density"] / 100, options["seed"])
    return [[u, i, int(matrix[u, i])] for u, i in mask.nonzero().tolist()]


TASKS = {task.id: task for task in (
    Task("classification", "二分类", "MLP", "根据两个特征，把记录分到两个类别。",
         "每行提供两个数值特征。两个示例分别把同号或异号判为类别 1。",
         FEATURES[:2], ("predicted_label", "probability_1"), FEATURES[:2] + ("label",), (
             Model("same-sign", "同号分类模型", "classification_project", "classification_model.pth", "ClassificationMLP"),
             Model("opposite-sign", "异号分类模型", "simple_dl_project", "model.pth", "SimpleMLP"),
         ), (ROWS, SEED), classification, labeled),
    Task("regression", "数值回归", "MLP", "用三个数值特征预测一个连续数值。",
         "每行提供三个数值特征。示例目标近似为 2×x1 − 1.5×x2 + 0.5×x3。",
         FEATURES[:3], ("prediction",), FEATURES[:3] + ("target",), (
             Model("default", "线性关系回归模型", "regression_project", "regression_model.pth", "RegressionMLP"),
         ), (ROWS, SEED), regression, labeled),
    Task("clustering", "数据聚类", "K-means", "根据已有聚类中心，把二维数据分到三组。",
         "每行提供两个数值。分组编号由模型决定，可能与生成数据中的标签顺序不同。",
         FEATURES[:2], ("cluster", "distance"), FEATURES[:2] + ("source_group",), (
             Model("default", "三个聚类中心", "clustering_project", "cluster_centers.pth", None),
         ), (Parameter("samples", "样本行数", 120, 3, 1998, 3), SEED), clustering, cluster_data),
    Task("dimensionality", "数据降维", "自编码器", "把四维特征压缩为两个坐标，便于查看分布。",
         "每行提供四个数值特征。结果包含二维坐标和四个重建值。",
         FEATURES, ("z1", "z2", "reconstructed_x1", "reconstructed_x2", "reconstructed_x3", "reconstructed_x4"), FEATURES, (
             Model("default", "四维到二维自编码器", "dimensionality_reduction_project", "autoencoder_model.pth", "AutoEncoder"),
         ), (ROWS, SEED), reduction, reduction_data),
    Task("anomaly", "异常检测", "自编码器", "计算重建误差，找出值得优先核查的记录。",
         "每行提供两个数值。重建误差越大越值得核查，结果可按误差排序。",
         FEATURES[:2], ("reconstruction_error",), FEATURES[:2] + ("is_injected_anomaly",), (
             Model("default", "异常检测自编码器", "anomaly_detection_project", "anomaly_model.pth", "AnomalyAutoEncoder"),
         ), (Parameter("normal", "正常样本数", 100, 1, 1000), Parameter("anomalies", "注入异常数", 20, 0, 1000), SEED), anomaly, anomaly_data),
    Task("sequence", "序列预测", "LSTM", "用最近 10 个观测值预测下一步。",
         "每行提供按时间从早到晚排列的 10 个数值，对应 t1 到 t10。",
         WINDOW, ("next_value",), WINDOW + ("actual_next",), (
             Model("default", "单变量序列模型", "sequence_modeling_project", "sequence_model.pth", "LSTMModel"),
         ), (ROWS, SEED), regression, sequence_data),
    Task("recommendation", "推荐评分", "矩阵分解", "预测已有用户对已知物品的评分。",
         "每行提供一对已有编号。用户范围为 0–49，物品范围为 0–29。",
         ("user_id", "item_id"), ("predicted_rating",), ("user_id", "item_id", "rating"), (
             Model("default", "用户–物品评分模型", "recommendation_project", "recommendation_model.pth", "MatrixFactorization", (50, 30)),
         ), (Parameter("density", "评分覆盖率（%）", 30, 1, 100), SEED), recommendation, rating_data),
)}


def model_for(task, model_id):
    if model_id is None:
        return task.models[0]
    for model in task.models:
        if model.id == model_id:
            return model
    raise InputError("请选择此任务下的模型。")


def catalog():
    return [{
        "id": task.id, "title": task.title, "method": task.method,
        "summary": task.summary, "input_hint": task.input_hint, "columns": task.columns,
        "output": task.output, "dataset": task.dataset,
        "models": [{"id": m.id, "name": m.name, "has_weights": (ROOT / m.module / m.weights).is_file()}
                   for m in task.models],
        "parameters": [vars(parameter) for parameter in task.parameters],
    } for task in TASKS.values()]


def generate(task, model, raw):
    if not isinstance(raw, dict):
        raise InputError("生成参数需要按字段提供。")
    options = {}
    for parameter in task.parameters:
        value = raw.get(parameter.key, parameter.default)
        if (type(value) is not int or not parameter.minimum <= value <= parameter.maximum
                or (value - parameter.minimum) % parameter.step):
            requirement = f"，且必须以 {parameter.minimum} 为起点按 {parameter.step} 递增" if parameter.step != 1 else ""
            raise InputError(f"{parameter.label}必须是 {parameter.minimum}–{parameter.maximum} 之间的整数{requirement}。")
        options[parameter.key] = value
    rows = task.generate(module_for(model), options)
    return {"columns": task.dataset, "rows": rows}


def read_csv(task, source):
    if not isinstance(source, str) or not source.strip():
        raise InputError("请填写数值、上传 CSV 或加载示例数据。")
    reader = csv.reader(io.StringIO(source.removeprefix("\ufeff")), strict=True)
    required = task.columns
    rows = []
    try:
        header = next(reader, None)
        if not header:
            raise InputError("CSV 第一行需要包含列名。")
        header = [column.strip() for column in header]
        if len(set(header)) != len(header) or "" in header:
            raise InputError("CSV 列名必须唯一且不能为空。")
        missing = [column for column in required if column not in header]
        if missing:
            raise InputError("CSV 缺少列 " + "、".join(missing))
        positions = [header.index(column) for column in required]
        for record in reader:
            line = reader.line_num
            if not record:
                continue
            if len(record) != len(header):
                raise InputError(f"CSV 第 {line} 行有 {len(record)} 列，表头有 {len(header)} 列。")
            if len(rows) >= MAX_ROWS:
                raise InputError(f"每次最多可运行 {MAX_ROWS} 行。")
            values = []
            for column, index in zip(required, positions):
                try:
                    value = float(record[index])
                except ValueError:
                    raise InputError(f"CSV 第 {line} 行的 {column} 需要填写数值。") from None
                if not math.isfinite(value) or abs(value) > 1e12:
                    raise InputError(f"CSV 第 {line} 行的 {column} 需要是绝对值不超过 1 万亿的有限数值。")
                if task.id == "recommendation":
                    maximum = 49 if column == "user_id" else 29
                    if not value.is_integer() or not 0 <= value <= maximum:
                        raise InputError(f"CSV 第 {line} 行的 {column} 需要是 0–{maximum} 之间的整数。")
                values.append(value)
            rows.append(values)
    except csv.Error as error:
        raise InputError(f"CSV 第 {reader.line_num} 行格式有误。") from error
    if not rows:
        raise InputError("CSV 需要至少一行数据。")
    return rows


def predict(task, model, source):
    rows = read_csv(task, source)
    import torch

    with LOAD_LOCK:
        network = load(model)
    with torch.inference_mode():
        predictions = task.predict(torch, network, torch.tensor(rows, dtype=torch.float32))
    if any(not math.isfinite(value) for row in predictions for value in row):
        raise InputError("模型返回了无效的数值。请检查输入范围或更换模型权重。")
    return {"columns": task.columns + task.output, "rows": [x + y for x, y in zip(rows, predictions, strict=True)]}
