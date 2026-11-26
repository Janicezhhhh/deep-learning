import math
from dataclasses import dataclass
from itertools import groupby

from .media import InputError


@dataclass(frozen=True)
class Reference:
    kind: str
    columns: tuple[str, ...]
    predictions: tuple[str, ...]
    maximum_label: int = 1

    def parse(self, value, column, line):
        if not value.strip():
            return None
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or abs(number) > 1e12:
            raise InputError(f"CSV 第 {line} 行的参考列 {column} 需要填写有限数值。")
        if self.kind != "error" and (not number.is_integer() or not 0 <= number <= self.maximum_label):
            raise InputError(f"CSV 第 {line} 行的参考列 {column} 需要填写 0–{self.maximum_label} 的整数标签。")
        return number


REFERENCES = {
    "classification": Reference("match", ("label",), ("predicted_label",)),
    "multiclass": Reference("match", ("label",), ("predicted_label",), 2),
    "multilabel": Reference("match", ("label_a", "label_b", "label_c"), ("label_a", "label_b", "label_c")),
    "regression": Reference("error", ("target",), ("prediction",)),
    "sequence": Reference("error", ("actual_next",), ("next_value",)),
    "multi-step": Reference("error", ("actual_1", "actual_2", "actual_3"), ("forecast_1", "forecast_2", "forecast_3")),
    "multi-variate": Reference("error", ("actual_next",), ("next_value",)),
    "recommendation": Reference("error", ("rating",), ("predicted_rating",)),
    "anomaly": Reference("ranking", ("is_injected_anomaly",), ("reconstruction_error",)),
    "series-anomaly": Reference("ranking", ("is_injected_anomaly",), ("absolute_error",)),
}


def metric(label, value, unit="number"):
    return {"label": label, "value": value, "unit": unit}


def evaluate_table(result, reference, answers):
    if not answers or not any(all(value is not None for value in row) for row in answers):
        return result
    indices = [result["columns"].index(name) for name in reference.predictions]
    matches, errors, scores = [], [], []
    rows = []
    detail = {"match": "label_match", "error": "row_mae", "ranking": None}[reference.kind]
    for i, (row, actual) in enumerate(zip(result["rows"], answers, strict=True), 1):
        estimate = [row[index] for index in indices]
        comparison = None
        if all(value is not None for value in actual):
            if reference.kind == "match":
                comparison = int(estimate == actual)
                matches.append(comparison)
            elif reference.kind == "error":
                difference = [abs(a - p) for a, p in zip(actual, estimate, strict=True)]
                comparison = sum(difference) / len(difference)
                errors.extend(difference)
            else:
                scores.append((estimate[0], actual[0]))
        rows.append([i, *row, *actual, *([comparison] if detail else [])])

    if reference.kind == "match":
        correct, count = sum(matches), len(matches)
        metrics = [metric("准确率" if len(indices) == 1 else "整行标签匹配率", correct / count, "percent"),
                   metric("预测正确", correct), metric("预测错误", count - correct)]
    elif reference.kind == "error":
        count = len(errors) // len(indices)
        metrics = [metric("平均绝对误差（MAE）", sum(errors) / len(errors)),
                   metric("均方根误差（RMSE）", math.sqrt(sum(e * e for e in errors) / len(errors)))]
    else:
        count = len(scores)
        anomalies = sum(label for _, label in scores)
        metrics = [metric("参考异常", int(anomalies)), metric("参考正常", int(count - anomalies))]
        if 0 < anomalies < count:
            # A tie between anomalous and normal scores counts for half a successful ordering.
            preceding_normals = wins = 0
            for _, group in groupby(sorted(scores), key=lambda pair: pair[0]):
                labels = [label for _, label in group]
                positives = sum(labels)
                negatives = len(labels) - positives
                wins += positives * (preceding_normals + negatives / 2)
                preceding_normals += negatives
            metrics.append(metric("排序面积（ROC AUC）", wins / (anomalies * (count - anomalies))))

    columns = ("sample_index", *result["columns"], *(f"reference_{name}" for name in reference.columns))
    return {"columns": columns + ((detail,) if detail else ()), "rows": rows,
            "evaluation": {"count": count, "metrics": metrics}}
