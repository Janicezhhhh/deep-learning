import io
import json
from dataclasses import dataclass
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .media import encode, table_csv


@dataclass
class MediaDataset:
    samples: list
    files: list
    annotations: list


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def zip_files(files):
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
    return encode(output.getvalue())


FIELD_DESCRIPTIONS = {
    "label": "参考类别编号",
    "target": "参考连续值",
    "actual_next": "下一步参考值",
    "observed": "待核查的实际观测值",
    "is_injected_anomaly": "是否注入异常（1 是，0 否）",
    "source_group": "生成时的原始簇编号",
    "user_id": "模型中的用户编号（0–49）",
    "item_id": "模型中的物品编号（0–29）",
    "rating": "参考评分（1–5）",
    "k": "请求的推荐结果数（1–10）",
}


def describe_field(column):
    if column.startswith("actual_"):
        return FIELD_DESCRIPTIONS.get(column, f"第 {column[7:]} 步参考值")
    if column.startswith("label_"):
        return f"参考标签 {column[-1].upper()}（1 有，0 无）"
    if column.startswith("aux"):
        return f"第 {column[3:]} 个时间点的辅助变量"
    if column.startswith("t") and column[1:].isdigit():
        return f"第 {column[1:]} 个历史时间点的数值"
    if column.startswith("x") and column[1:].isdigit():
        return f"数值特征 {column[1:]}"
    return FIELD_DESCRIPTIONS.get(column, column)


def fields_for(task, annotations):
    if task.dataset_type == "table":
        return [{"name": column, "type": "number", "role": "input" if column in task.columns else "reference",
                 "description": describe_field(column)} for column in task.dataset]
    descriptions = {
        "image": "PNG 文件的相对路径", "audio": "PCM WAV 文件的相对路径", "mask": "灰度 PNG 掩膜的相对路径",
        "objects": "可见物体标注，检测框坐标为 [左, 上, 右, 下]，右边界和下边界不含端点",
        "label": "参考数字或类别标签", "transcript": "生成音频对应的参考转写",
        "fields": "文本输入字段", "target": "参考标签、答案或译文", "name": "样本名称",
    }
    return [{"name": key, "description": descriptions[key]} for key in annotations[0] if key in descriptions] if annotations else []


def export_dataset(task, model, options, batches, test_percent):
    manifest = {
        "schema_version": 1,
        "task": {"id": task.id, "name": task.title, "input_type": task.input_type},
        "model": {"id": model.id, "name": model.name},
        "seed": options["seed"], "parameters": options,
        "parameter_labels": {parameter.key: parameter.label for parameter in task.parameters},
        "test_percent": test_percent, "fields": [], "partitions": [],
    }
    if task.dataset_type == "table":
        result = {"columns": task.dataset, "rows": []}
    else:
        result = {"samples": []}
    files = []
    for split, settings, dataset in batches:
        if task.dataset_type == "table":
            partition_files = [("data.csv", table_csv(task.dataset, dataset))]
            annotations = []
            result["rows"].extend(dataset)
            count = len(dataset)
        else:
            partition_files = [*dataset.files, ("annotations.json", json_bytes(dataset.annotations))]
            annotations = dataset.annotations
            result["samples"].extend({**sample, "split": split} if split != "all" else sample
                                     for sample in dataset.samples)
            count = len(dataset.samples)
        if not manifest["fields"]:
            manifest["fields"] = fields_for(task, annotations)
        prefix = "" if split == "all" else f"{split}/"
        partition_files = [(prefix + name, data) for name, data in partition_files]
        files.extend(partition_files)
        manifest["partitions"].append({"name": split, "record_count": count, "parameters": settings,
                                       "files": [name for name, _ in partition_files]})
    return {**result, "manifest": manifest, "archive": zip_files([("manifest.json", json_bytes(manifest)), *files])}
