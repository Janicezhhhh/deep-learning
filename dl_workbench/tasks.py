import csv
import importlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS = 2000
FEATURES = ("x1", "x2", "x3", "x4")
WINDOW = tuple(f"t{i}" for i in range(1, 11))
PAIRED_WINDOW = tuple(f"{feature}{i}" for i in range(1, 11) for feature in ("t", "aux"))
RATING_IDS = (("user_id", 0, 49), ("item_id", 0, 29))
LOAD_LOCK = Lock()
GROUPS = {
    "classify": "判断类别",
    "forecast": "预测数值",
    "explore": "探索数据",
    "recommend": "推荐与评分",
}
INPUT_TYPES = {"table": "数值表格", "series": "时间序列"}


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
    group: str = field(kw_only=True)
    input_type: str = "table"
    result_type: str = "table"
    integer_ranges: tuple[tuple[str, int, int], ...] = ()
    expands_rows: bool = False

    @property
    def result_columns(self):
        return (("query",) if self.expands_rows else ()) + self.columns + self.output


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


def multiclass(torch, model, rows):
    probability = model(rows).softmax(dim=1)
    return torch.column_stack((probability.argmax(dim=1), probability)).tolist()


def multilabel(torch, model, rows):
    probability = model(rows).sigmoid()
    return torch.cat((probability >= 0.5, probability), dim=1).tolist()


def forecast_error(torch, model, rows):
    prediction = model(rows[:, :10])[:, :1]
    return torch.cat((prediction, (rows[:, 10:] - prediction).abs()), dim=1).tolist()


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


def top_items(torch, model, rows):
    items = torch.arange(30)
    user_ids = rows[:, 0].long()
    ratings = model(user_ids.repeat_interleave(len(items)), items.repeat(len(rows))).reshape(len(rows), -1)
    ranking = ratings.argsort(dim=1, descending=True, stable=True).tolist()
    results = []
    for row_number, ((user, k), ranked, scores) in enumerate(zip(rows.int().tolist(), ranking, ratings.tolist()), 1):
        for rank, item in enumerate(ranked[:k], 1):
            results.append([row_number, user, k, rank, item, scores[item]])
    return results


def labeled(module, options):
    x, y = module.make_data(options["samples"], options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), y.flatten().tolist())]


def numeric_generator(name, *args):
    def generate_numeric(module, options):
        x, y = getattr(module, name)(options["samples"], options["seed"], *args)
        return [features + outputs for features, outputs in zip(x.tolist(), y.reshape(len(x), -1).tolist(), strict=True)]
    return generate_numeric


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


def rank_data(module, options):
    from random import Random

    rng = Random(options["seed"])
    return [[user, 5] for user in rng.sample(range(50), options["samples"])]


MULTI = Model("default", "三类别 MLP", "classic_tabular", "multiclass.pth", "MulticlassMLP")
LABELS = Model("default", "独立标签 MLP", "classic_tabular", "multilabel.pth", "MultilabelMLP")
MULTISTEP = Model("default", "三步序列 LSTM", "classic_tabular", "multistep.pth", "ForecastLSTM")
MULTIVARIATE = Model("default", "双变量 LSTM", "classic_tabular", "multivariate.pth", "ForecastLSTM", (2, 1))
RATINGS = Model("default", "用户–物品评分模型", "recommendation_project", "recommendation_model.pth", "MatrixFactorization", (50, 30))
HORIZONS = ("forecast_1", "forecast_2", "forecast_3")
LABEL_COLUMNS = ("label_a", "label_b", "label_c")


TASKS = {task.id: task for task in (
    Task("classification", "二分类", "MLP", "根据两个特征，把记录分到两个类别。",
         "每行提供两个数值特征。两个示例分别把同号或异号判为类别 1。",
         FEATURES[:2], ("predicted_label", "probability_1"), FEATURES[:2] + ("label",), (
             Model("same-sign", "同号分类模型", "classification_project", "classification_model.pth", "ClassificationMLP"),
             Model("opposite-sign", "异号分类模型", "simple_dl_project", "model.pth", "SimpleMLP"),
         ), (ROWS, SEED), classification, labeled, group="classify"),
    Task("multiclass", "表格多分类", "MLP", "从三个类别中为每条记录选一个类别。",
         "每行提供两个数值。示例模型识别三组数值样本，类别为 0、1 或 2。",
         FEATURES[:2], ("predicted_label", "probability_0", "probability_1", "probability_2"),
         FEATURES[:2] + ("label",), (MULTI,), (ROWS, SEED), multiclass, numeric_generator("multiclass_data"), group="classify"),
    Task("multilabel", "表格多标签分类", "MLP", "同一条记录可以同时获得多个标签。",
         "每行提供三个数值。模型独立评估 A、B、C 三种标签，概率达到 0.5 时标为 1。",
         FEATURES[:3], LABEL_COLUMNS + ("probability_a", "probability_b", "probability_c"),
         FEATURES[:3] + LABEL_COLUMNS, (LABELS,), (ROWS, SEED), multilabel, numeric_generator("multilabel_data"), group="classify"),
    Task("regression", "数值回归", "MLP", "用三个数值特征预测一个连续数值。",
         "每行提供三个数值特征。示例目标近似为 2×x1 − 1.5×x2 + 0.5×x3。",
         FEATURES[:3], ("prediction",), FEATURES[:3] + ("target",), (
             Model("default", "线性关系回归模型", "regression_project", "regression_model.pth", "RegressionMLP"),
         ), (ROWS, SEED), regression, labeled, group="forecast"),
    Task("clustering", "数据聚类", "K-means", "根据已有聚类中心，把二维数据分到三组。",
         "每行提供两个数值。分组编号由模型决定，可能与生成数据中的标签顺序不同。",
         FEATURES[:2], ("cluster", "distance"), FEATURES[:2] + ("source_group",), (
             Model("default", "三个聚类中心", "clustering_project", "cluster_centers.pth", None),
         ), (Parameter("samples", "样本行数", 120, 3, 1998, 3), SEED), clustering, cluster_data,
         group="explore", result_type="cluster-scatter"),
    Task("dimensionality", "数据降维", "自编码器", "把四维特征压缩为两个坐标，便于查看分布。",
         "每行提供四个数值特征。结果包含二维坐标和四个重建值。",
         FEATURES, ("z1", "z2", "reconstructed_x1", "reconstructed_x2", "reconstructed_x3", "reconstructed_x4"), FEATURES, (
             Model("default", "四维到二维自编码器", "dimensionality_reduction_project", "autoencoder_model.pth", "AutoEncoder"),
         ), (ROWS, SEED), reduction, reduction_data, group="explore", result_type="embedding-scatter"),
    Task("anomaly", "异常检测", "自编码器", "计算重建误差，找出值得优先核查的记录。",
         "每行提供两个数值。重建误差越大越值得核查，结果可按误差排序。",
         FEATURES[:2], ("reconstruction_error",), FEATURES[:2] + ("is_injected_anomaly",), (
             Model("default", "异常检测自编码器", "anomaly_detection_project", "anomaly_model.pth", "AnomalyAutoEncoder"),
         ), (Parameter("normal", "正常样本数", 100, 1, 1000), Parameter("anomalies", "注入异常数", 20, 0, 1000), SEED), anomaly, anomaly_data,
         group="explore", result_type="anomaly-scores"),
    Task("sequence", "序列预测", "LSTM", "用最近 10 个观测值预测下一步。",
         "每行提供按时间从早到晚排列的 10 个数值，对应 t1 到 t10。",
         WINDOW, ("next_value",), WINDOW + ("actual_next",), (
             Model("default", "单变量序列模型", "sequence_modeling_project", "sequence_model.pth", "LSTMModel"),
         ), (ROWS, SEED), regression, sequence_data, group="forecast", input_type="series"),
    Task("multi-step", "多步序列预测", "LSTM", "根据最近 10 个值，一次预测接下来的 3 步。",
         "每行提供从早到晚的 10 个数值。权重基于带噪声的周期序列。",
         WINDOW, HORIZONS, WINDOW + ("actual_1", "actual_2", "actual_3"), (MULTISTEP,), (ROWS, SEED),
         regression, numeric_generator("forecast_data"), group="forecast", input_type="series"),
    Task("multi-variate", "多变量序列预测", "LSTM", "结合目标序列与辅助变量，预测目标的下一步。",
         "每行提供 10 个时间点，按 t1、aux1、t2、aux2 依次排列。辅助变量需要与目标时间对齐。",
         PAIRED_WINDOW, ("next_value",), PAIRED_WINDOW + ("actual_next",), (MULTIVARIATE,), (ROWS, SEED),
         regression, numeric_generator("forecast_data", 2, 1), group="forecast", input_type="series"),
    Task("series-anomaly", "时间序列异常检测", "LSTM", "比较下一步观测与预测，找出偏离趋势的点。",
         "每行提供 10 个历史值和待核查的观测值。预测偏差越大越值得核查。",
         WINDOW + ("observed",), ("next_value", "absolute_error"), WINDOW + ("observed", "is_injected_anomaly"),
         (MULTISTEP,), (ROWS, SEED), forecast_error, numeric_generator("temporal_anomaly_data"),
         group="explore", input_type="series", result_type="forecast-errors"),
    Task("recommendation", "推荐评分", "矩阵分解", "预测已有用户对已知物品的评分。",
         "每行提供一对已有编号。用户范围为 0–49，物品范围为 0–29。",
         ("user_id", "item_id"), ("predicted_rating",), ("user_id", "item_id", "rating"), (RATINGS,),
         (Parameter("density", "评分覆盖率（%）", 30, 1, 100), SEED), recommendation, rating_data,
         group="recommend", integer_ranges=RATING_IDS),
    Task("top-k", "Top-K 推荐", "矩阵分解", "为每位用户列出预测评分最高的物品。",
         "输入用户编号 0–49 和想要的结果数 k（1–10）。候选为模型已知的 30 件物品。",
         ("user_id", "k"), ("rank", "item_id", "predicted_rating"), ("user_id", "k"), (RATINGS,),
         (Parameter("samples", "用户数", 12, 1, 50), SEED), top_items, rank_data, group="recommend",
         integer_ranges=(("user_id", 0, 49), ("k", 1, 10)), expands_rows=True),
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
        "group": task.group, "input_type": task.input_type, "result_type": task.result_type,
        "summary": task.summary, "input_hint": task.input_hint, "columns": task.columns,
        "output": task.output, "result_columns": task.result_columns, "dataset": task.dataset,
        "integer_ranges": {column: [low, high] for column, low, high in task.integer_ranges},
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
    integer_ranges = {column: (low, high) for column, low, high in task.integer_ranges}
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
                if column in integer_ranges:
                    low, high = integer_ranges[column]
                    if not value.is_integer() or not low <= value <= high:
                        raise InputError(f"CSV 第 {line} 行的 {column} 需要是 {low}–{high} 之间的整数。")
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
    if task.expands_rows:
        return {"columns": task.result_columns, "rows": predictions}
    return {"columns": task.result_columns, "rows": [x + y for x, y in zip(rows, predictions, strict=True)]}
