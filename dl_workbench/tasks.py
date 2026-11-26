import csv
import importlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .datasets import export_dataset
from .evaluation import REFERENCES, evaluate_table
from .language_tasks import SPEECH_NOISE_RANGE, SPEECH_SPEED_RANGE, audio_predict, sample_dataset, text_predict
from .media import InputError
from .model_assets import available
from .vision_tasks import digit_predict, image_dataset, scene_predict


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
    "understand": "识别与提取",
    "transform": "转换与生成",
}
INPUT_TYPES = {"table": "数值表格", "series": "时间序列", "image": "图片", "text": "文本", "audio": "音频"}


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    default: int
    minimum: int
    maximum: int
    step: int = 1
    description: str = ""


ROWS = Parameter("samples", "样本行数", 120, 1, MAX_ROWS)
SEED = Parameter("seed", "随机种子", 42, 0, 2**31 - 1)
FEATURE_NOISE = Parameter("noise_percent", "特征扰动（%）", 0, 0, 200, description="0 为原始特征。提高扰动会保留原标签并改变输入。")
BASE_NOISE = Parameter("noise_percent", "噪声倍率（%）", 100, 0, 300, description="100 为原始噪声，0 关闭随机扰动。")
DISTANCE = Parameter("distance_percent", "类间距离倍率（%）", 100, 25, 200, description="数值越小，各类别的样本越容易重叠。")
SERIES_CONTROLS = (
    Parameter("trend_percent", "每十步趋势（%）", 0, -200, 200, description="相对于单位振幅，负值产生下行趋势。"),
    Parameter("period_percent", "周期倍率（%）", 100, 50, 200, description="100 为原周期，200 表示周期加倍。"),
    Parameter("jump_percent", "突变幅度（%）", 0, -200, 200, description="相对于单位振幅，在序列中途产生一次水平跳变。"),
)


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    module: str
    weights: str
    class_name: str | None
    arguments: tuple = ()
    asset_key: str = ""

    @property
    def has_weights(self):
        if self.asset_key:
            return available(self.asset_key)
        return (ROOT / self.module / self.weights).is_file()


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
    predict: Callable | None
    generate: Callable
    group: str = field(kw_only=True)
    input_type: str = "table"
    result_type: str = "table"
    integer_ranges: tuple[tuple[str, int, int], ...] = ()
    expands_rows: bool = False
    request_predict: Callable | None = None
    dataset_type: str = "table"
    fields: tuple[tuple[str, str], ...] = ()
    batch_limit: int = 0
    predict_parameters: tuple[Parameter, ...] = ()

    @property
    def result_columns(self):
        return (("query",) if self.expands_rows else ()) + self.columns + self.output

    @property
    def split_parameter(self):
        labeled_table = self.dataset_type == "table" and bool(set(self.dataset) - set(self.columns))
        if labeled_table or self.input_type in ("image", "audio") or self.id == "speech":
            return next((parameter for parameter in self.parameters if parameter.key == "samples"), None)
        return None


def module_for(model):
    return importlib.import_module(f"{model.module}.train")


@lru_cache(maxsize=16)
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
    controls = {"noise_percent": options["noise_percent"]} if "noise_percent" in options else {}
    x, y = module.make_data(options["samples"], options["seed"], **controls)
    return [features + [label] for features, label in zip(x.tolist(), y.flatten().tolist())]


def binary_data(module, options):
    import torch

    count, seed = options["samples"], options["seed"]
    x, y = module.make_data(count, seed)
    labels = y.flatten()
    generator = torch.Generator().manual_seed(seed ^ 197)
    if options["positive_percent"] != 50:
        positives = round(count * options["positive_percent"] / 100)
        requested = torch.zeros_like(labels)
        requested[torch.randperm(count, generator=generator)[:positives]] = 1
        x[requested != labels, 1] *= -1
        labels = requested
    if options["noise_percent"]:
        x += (options["noise_percent"] / 100) * torch.randn(x.shape, generator=generator)
    return [features + [label] for features, label in zip(x.tolist(), labels.tolist(), strict=True)]


def numeric_generator(name, *args, controls=()):
    def generate_numeric(module, options):
        x, y = getattr(module, name)(options["samples"], options["seed"], *args,
                                     **{parameter.key: options[parameter.key] for parameter in controls})
        return [features + outputs for features, outputs in zip(x.tolist(), y.reshape(len(x), -1).tolist(), strict=True)]
    return generate_numeric


def cluster_data(module, options):
    x, y, _ = module.make_data(options["samples"], options["seed"], noise_percent=options["noise_percent"],
                               distance_percent=options["distance_percent"])
    return [features + [label] for features, label in zip(x.tolist(), y.tolist())]


def reduction_data(module, options):
    return module.make_data(options["samples"], options["seed"]).tolist()


def anomaly_data(module, options):
    x, labels = module.make_data(options["normal"], options["anomalies"], options["seed"])
    return [features + [label] for features, label in zip(x.tolist(), labels.tolist())]


def sequence_data(module, options):
    x, y = module.make_sequence_data(options["samples"] + len(WINDOW) + 1, len(WINDOW), options["seed"],
                                     period_steps=150,
                                     **{parameter.key: options[parameter.key] for parameter in SERIES_CONTROLS})
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
BINARY_CONTROLS = (Parameter("positive_percent", "类别 1 比例（%）", 50, 0, 100, description="50 保留原始抽样，其他值按指定比例分配。"), FEATURE_NOISE)
MULTI_CONTROLS = (Parameter("class_zero_percent", "类别 0 抽样权重（%）", 33, 0, 100, description="剩余抽样权重均分给类别 1 和类别 2。"), BASE_NOISE, DISTANCE)
DIGIT = Model("default", "手写数字 CNN", "classic_vision", "digit_cnn.pth", "DigitCNN")
SHAPES = Model("default", "几何图形分割 CNN", "classic_vision", "shape_net.pth", "ShapeNet")
IMAGES = Parameter("samples", "图片数", 6, 1, 12)
IMAGE_VARIATIONS = (
    Parameter("rotation_degrees", "额外旋转（°）", 0, -45, 45, description="正值逆时针，负值顺时针。旋转后整张图适配画布。"),
    Parameter("size_percent", "物体大小倍率（%）", 100, 50, 130, description="100 为原尺寸。物体大小受画布限制。"),
    Parameter("occlusion_percent", "遮挡比例（%）", 0, 0, 45, description="每个物体的前景像素被遮住的近似比例。"),
)
DIGITS = (IMAGES, Parameter("digit", "数字（−1 为随机）", -1, -1, 9), SEED, *IMAGE_VARIATIONS)
SCENES = (IMAGES, Parameter("objects", "每图物体数", 4, 1, 9), SEED, *IMAGE_VARIATIONS)
BOX_FIELDS = ("instance", "class_name", "probability", "x_min", "y_min", "x_max", "y_max")
TEXT_SAMPLES = (Parameter("samples", "文本条数", 6, 1, 12), SEED)
AUDIO_VARIATIONS = (
    Parameter("speed_percent", "语速倍率（%）", 100, *SPEECH_SPEED_RANGE, description="100 为正常语速。"),
    Parameter("noise_percent", "背景噪声（%）", 0, *SPEECH_NOISE_RANGE, description="0 无附加噪声，100 表示噪声与语音的均方根幅度相当。"),
)
AUDIO_SAMPLES = (Parameter("samples", "音频条数", 3, 1, 6), SEED, *AUDIO_VARIATIONS)


def language_task(key, title, method, summary, hint, output, group, *, fields=(("text", "英文文本"),),
                  input_type="text", result_type="table"):
    model = Model("default", method, "", "", None, asset_key=key)
    audio = input_type == "audio"
    return Task(key, title, method, summary, hint, (), output, (), (model,), AUDIO_SAMPLES if audio or key == "speech" else TEXT_SAMPLES,
                None, sample_dataset(key), group=group, input_type=input_type, result_type=result_type,
                request_predict=audio_predict if audio else text_predict, dataset_type="samples", fields=() if audio else fields,
                batch_limit=64 if key == "sentiment" else 0, predict_parameters=AUDIO_VARIATIONS if key == "speech" else ())


TASKS = {task.id: task for task in (
    Task("classification", "二分类", "MLP", "根据两个特征，把记录分到两个类别。",
         "每行提供两个数值特征。两个示例分别把同号或异号判为类别 1。",
         FEATURES[:2], ("predicted_label", "probability_1"), FEATURES[:2] + ("label",), (
             Model("same-sign", "同号分类模型", "classification_project", "classification_model.pth", "ClassificationMLP"),
             Model("opposite-sign", "异号分类模型", "simple_dl_project", "model.pth", "SimpleMLP"),
         ), (ROWS, SEED, *BINARY_CONTROLS), classification, binary_data, group="classify"),
    Task("multiclass", "表格多分类", "MLP", "从三个类别中为每条记录选一个类别。",
         "每行提供两个数值。示例模型识别三组数值样本，类别为 0、1 或 2。",
         FEATURES[:2], ("predicted_label", "probability_0", "probability_1", "probability_2"),
         FEATURES[:2] + ("label",), (MULTI,), (ROWS, SEED, *MULTI_CONTROLS), multiclass,
         numeric_generator("multiclass_data", controls=MULTI_CONTROLS), group="classify"),
    Task("multilabel", "表格多标签分类", "MLP", "同一条记录可以同时获得多个标签。",
         "每行提供三个数值。模型独立评估 A、B、C 三种标签，概率达到 0.5 时标为 1。",
         FEATURES[:3], LABEL_COLUMNS + ("probability_a", "probability_b", "probability_c"),
         FEATURES[:3] + LABEL_COLUMNS, (LABELS,), (ROWS, SEED, FEATURE_NOISE), multilabel,
         numeric_generator("multilabel_data", controls=(FEATURE_NOISE,)), group="classify"),
    Task("digit", "手写数字识别", "CNN", "识别图片中的一个手写数字，查看 0–9 的类别概率。",
         "上传包含单个数字、背景干净的 PNG 或 JPEG。模型依据 MNIST 与合成数字训练。",
         (), ("predicted_label", "probability"), (), (DIGIT,), DIGITS, None, image_dataset("digit"),
         group="classify", input_type="image", request_predict=digit_predict, dataset_type="samples"),
    Task("object-detection", "目标检测", "CNN", "定位圆形、方形和三角形，标出检测框。",
         "用配套样本体验。示例模型面向深色背景的几何图形，框坐标按上传图片的像素计算。",
         (), BOX_FIELDS, (), (SHAPES,), SCENES, None, image_dataset("detection"),
         group="understand", input_type="image", result_type="image-preview", request_predict=scene_predict("detection"), dataset_type="samples"),
    Task("semantic-segmentation", "语义分割", "CNN", "为每个像素标注几何类别，查看分割掩膜。",
         "用配套样本体验。示例类别为背景、圆形、方形和三角形。",
         (), ("class_name", "pixel_fraction"), (), (SHAPES,), SCENES, None, image_dataset("semantic"),
         group="understand", input_type="image", result_type="image-preview", request_predict=scene_predict("semantic"), dataset_type="samples"),
    Task("instance-segmentation", "实例分割", "CNN 与连通域", "把相互分开的同类几何物体标成独立实例。",
         "用配套样本体验。同类别物体需要彼此分开，以便模型掩膜的连通域分出不同实例。",
         (), BOX_FIELDS, (), (SHAPES,), SCENES, None, image_dataset("instance"),
         group="understand", input_type="image", result_type="image-preview", request_predict=scene_predict("instance"), dataset_type="samples"),
    Task("regression", "数值回归", "MLP", "用三个数值特征预测一个连续数值。",
         "每行提供三个数值特征。示例目标近似为 2×x1 − 1.5×x2 + 0.5×x3。",
         FEATURES[:3], ("prediction",), FEATURES[:3] + ("target",), (
             Model("default", "线性关系回归模型", "regression_project", "regression_model.pth", "RegressionMLP"),
         ), (ROWS, SEED, BASE_NOISE), regression, labeled, group="forecast"),
    Task("clustering", "数据聚类", "K-means", "根据已有聚类中心，把二维数据分到三组。",
         "每行提供两个数值。分组编号由模型决定，可能与生成数据中的标签顺序不同。",
         FEATURES[:2], ("cluster", "distance"), FEATURES[:2] + ("source_group",), (
             Model("default", "三个聚类中心", "clustering_project", "cluster_centers.pth", None),
         ), (Parameter("samples", "样本行数", 120, 3, 1998, 3), SEED, BASE_NOISE, DISTANCE), clustering, cluster_data,
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
         ), (ROWS, SEED, *SERIES_CONTROLS), regression, sequence_data, group="forecast", input_type="series"),
    Task("multi-step", "多步序列预测", "LSTM", "根据最近 10 个值，一次预测接下来的 3 步。",
         "每行提供从早到晚的 10 个数值。权重基于带噪声的周期序列。",
         WINDOW, HORIZONS, WINDOW + ("actual_1", "actual_2", "actual_3"), (MULTISTEP,), (ROWS, SEED, *SERIES_CONTROLS),
         regression, numeric_generator("forecast_data", controls=SERIES_CONTROLS), group="forecast", input_type="series"),
    Task("multi-variate", "多变量序列预测", "LSTM", "结合目标序列与辅助变量，预测目标的下一步。",
         "每行提供 10 个时间点，按 t1、aux1、t2、aux2 依次排列。辅助变量需要与目标时间对齐。",
         PAIRED_WINDOW, ("next_value",), PAIRED_WINDOW + ("actual_next",), (MULTIVARIATE,), (ROWS, SEED, *SERIES_CONTROLS),
         regression, numeric_generator("forecast_data", 2, 1, controls=SERIES_CONTROLS), group="forecast", input_type="series"),
    Task("series-anomaly", "时间序列异常检测", "LSTM", "比较下一步观测与预测，找出偏离趋势的点。",
         "每行提供 10 个历史值和待核查的观测值。预测偏差越大越值得核查。",
         WINDOW + ("observed",), ("next_value", "absolute_error"), WINDOW + ("observed", "is_injected_anomaly"),
         (MULTISTEP,), (ROWS, SEED, *SERIES_CONTROLS), forecast_error, numeric_generator("temporal_anomaly_data", controls=SERIES_CONTROLS),
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
    language_task("sentiment", "文本分类", "DistilBERT", "判断英文评论的情感倾向。",
                  "粘贴一段英文评论，也可以上传带 text 列的 CSV 进行批量分类。结果给出正向和负向概率。",
                  ("class_name", "probability"), "classify"),
    language_task("entities", "命名实体识别", "DistilBERT NER", "从英文段落中提取人物、机构、地点等实体。",
                  "粘贴英文段落。结果显示原文片段、类别和字符位置，结束位置不含端点。",
                  ("entity", "entity_type", "start", "end", "probability"), "understand"),
    language_task("translation", "机器翻译", "MarianMT", "将英文翻译为简体中文。",
                  "输入一段简短英文。适合句子或短段落。", ("translation",), "transform"),
    language_task("summary", "文本摘要", "FLAN-T5", "将英文段落浓缩成一两句话。",
                  "输入一段英文正文。模型根据提供的文本生成英文摘要。", ("summary",), "transform"),
    language_task("question", "阅读理解问答", "DistilBERT SQuAD", "从给定英文段落中找出问题的答案。",
                  "填写参考段落和一个答案能在段落中找到的英文问题。", ("answer", "start", "end", "probability"),
                  "understand", fields=(("context", "英文参考段落"), ("question", "英文问题"))),
    language_task("generation", "文本生成", "SmolLM2 135M", "根据英文提示生成一段短文本。",
                  "使用英文给出明确指令，例如写三句简短介绍。", ("generated_text",), "transform",
                  fields=(("text", "英文提示"),)),
    language_task("keywords", "音频分类", "Wav2Vec2", "识别音频中的英文命令词。",
                  "上传最多 3 秒的 PCM WAV，建议只含一个命令词。生成语音样本需要 speech 权重。",
                  ("class_name", "probability"), "classify", input_type="audio"),
    language_task("transcription", "语音识别", "Whisper Tiny", "把英文语音转成文本。",
                  "上传最长 30 秒的 PCM WAV。生成示例语音需要 speech 权重。",
                  ("transcript",), "understand", input_type="audio"),
    language_task("speech", "语音合成", "MMS-TTS", "把英文文本合成为可播放的 WAV。",
                  "输入最多 500 个字符的英文。使用模型内置音色。", (), "transform", result_type="audio-preview"),
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
        "dataset_type": task.dataset_type, "fields": task.fields, "batch_limit": task.batch_limit,
        "summary": task.summary, "input_hint": task.input_hint, "columns": task.columns,
        "output": task.output, "result_columns": task.result_columns, "dataset": task.dataset,
        "reference_columns": REFERENCES[task.id].columns if task.id in REFERENCES else (),
        "integer_ranges": {column: [low, high] for column, low, high in task.integer_ranges},
        "models": [{"id": m.id, "name": m.name, "has_weights": m.has_weights, "asset_key": m.asset_key}
                   for m in task.models],
        "parameters": [vars(parameter) for parameter in task.parameters],
        "supports_split": task.split_parameter is not None,
        "predict_parameters": [vars(parameter) for parameter in task.predict_parameters],
    } for task in TASKS.values()]


def generation_partitions(task, options, test_percent):
    if type(test_percent) is not int or not 0 <= test_percent <= 90:
        raise InputError("测试集比例必须是 0–90 之间的整数。")
    if not test_percent:
        return [("all", options)]
    parameter = task.split_parameter
    if parameter is None:
        raise InputError("此任务采用独立的数据设置，当前仅可导出完整样本。")
    count = options[parameter.key]
    group_count = count // parameter.step
    min_groups = math.ceil(parameter.minimum / parameter.step)
    if group_count < min_groups * 2:
        raise InputError(f"生成训练集和测试集至少需要 {min_groups * 2 * parameter.step} 个样本。")
    test_count = min(group_count - min_groups, max(min_groups, round(group_count * test_percent / 100))) * parameter.step
    return [("train", {**options, parameter.key: count - test_count}),
            ("test", {**options, parameter.key: test_count, "seed": options["seed"] ^ 0x5BD1E995})]


def generate(task, model, raw, test_percent=0):
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
    partitions = generation_partitions(task, options, test_percent)
    module = module_for(model) if model.module else None
    batches = [(name, settings, task.generate(module, settings)) for name, settings in partitions]
    return export_dataset(task, model, options, batches, test_percent)


def read_csv(task, source):
    if not isinstance(source, str) or not source.strip():
        raise InputError("请填写数值、上传 CSV 或加载示例数据。")
    reader = csv.reader(io.StringIO(source.removeprefix("\ufeff")), strict=True)
    required = task.columns
    rows = []
    answers = []
    reference = REFERENCES.get(task.id)
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
        reference_positions = ([header.index(column) for column in reference.columns]
                               if reference and all(column in header for column in reference.columns) else [])
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
            if reference_positions:
                answers.append([reference.parse(record[index], column, line)
                                for index, column in zip(reference_positions, reference.columns, strict=True)])
    except csv.Error as error:
        raise InputError(f"CSV 第 {reader.line_num} 行格式有误。") from error
    if not rows:
        raise InputError("CSV 需要至少一行数据。")
    return rows, answers


def predict(task, model, source):
    rows, answers = read_csv(task, source)
    import torch

    with LOAD_LOCK:
        network = load(model)
    with torch.inference_mode():
        predictions = task.predict(torch, network, torch.tensor(rows, dtype=torch.float32))
    if any(not math.isfinite(value) for row in predictions for value in row):
        raise InputError("模型返回了无效的数值。请检查输入范围或更换模型权重。")
    if task.expands_rows:
        return {"columns": task.result_columns, "rows": predictions}
    result = {"columns": task.result_columns, "rows": [x + y for x, y in zip(rows, predictions, strict=True)]}
    return evaluate_table(result, REFERENCES.get(task.id), answers)


def predict_request(task, model, payload):
    if task.request_predict is None:
        return predict(task, model, payload.get("csv"))
    if model.asset_key:
        return task.request_predict(model.asset_key, payload)
    with LOAD_LOCK:
        network = load(model)
    return task.request_predict(network, payload)
