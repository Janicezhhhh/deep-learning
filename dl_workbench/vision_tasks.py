import io
from collections import deque
from random import Random

from .evaluation import metric
from .media import InputError, binary_input, dataset_zip, encode


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_image(payload):
    from PIL import Image, ImageOps, UnidentifiedImageError

    raw = binary_input(payload.get("image"), "图片")
    try:
        with Image.open(io.BytesIO(raw), formats=("PNG", "JPEG")) as image:
            if max(image.size) > 4096 or image.width * image.height > 8_000_000:
                raise InputError("图片最大边长为 4096 像素，总像素不能超过 800 万。")
            image = ImageOps.exif_transpose(image)
            if "A" in image.getbands() or "transparency" in image.info:
                background = Image.new("RGBA", image.size, "white")
                return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
            return image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise InputError("无法读取图片。请选择一张有效的 PNG 或 JPEG。") from error


def digit_input(image):
    from PIL import Image, ImageOps
    from statistics import median

    gray = ImageOps.grayscale(image)
    w, h = gray.size
    border = list(gray.crop((0, 0, w, 1)).getdata()) + list(gray.crop((0, h - 1, w, h)).getdata())
    if median(border) > 127:
        gray = ImageOps.invert(gray)
    ink = gray.point(lambda pixel: 255 if pixel > 60 else 0).getbbox()
    if not ink or gray.getextrema()[1] - gray.getextrema()[0] < 40:
        raise InputError("图片里的笔画和背景差别太小。请选择一张清晰的单个数字图片。")
    if gray.size == (28, 28):
        return gray
    cropped = gray.crop(ink)
    cropped.thumbnail((21, 21), Image.Resampling.LANCZOS)
    if max(cropped.size) < 21:
        factor = 21 / max(cropped.size)
        cropped = cropped.resize(tuple(max(1, round(side * factor)) for side in cropped.size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(cropped, ((28 - cropped.width) // 2, (28 - cropped.height) // 2))
    return canvas


def digit_predict(network, payload):
    import torch
    from classic_vision.train import image_tensor

    image = digit_input(load_image(payload))
    with torch.inference_mode():
        probabilities = network(image_tensor(image)[None]).softmax(dim=1)[0].tolist()
    rows = sorted(enumerate(probabilities), key=lambda row: row[1], reverse=True)
    result = {"columns": ("predicted_label", "probability"), "rows": rows}
    reference = payload.get("reference")
    if reference is not None:
        label = reference.get("label") if isinstance(reference, dict) else None
        if type(label) is not int or not 0 <= label <= 9:
            raise InputError("参考数字必须是 0–9 的整数。")
        prediction = rows[0][0]
        result["evaluation"] = {"count": 1, "metrics": [metric("参考数字", label), metric("首选预测", prediction),
                                                        metric("识别正确", int(label == prediction), "match")]}
        result["columns"] += ("reference_label",)
        result["rows"] = [[*row, label] for row in rows]
    return result


def connected_objects(classes):
    import numpy as np

    height, width = classes.shape
    for label in range(1, 4):
        remaining = set(zip(*np.nonzero(classes == label)))
        while remaining:
            point = remaining.pop()
            queue = deque([point])
            pixels = [point]
            while queue:
                y, x = queue.popleft()
                for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        pixels.append(neighbor)
            if len(pixels) >= 20:
                ys, xs = np.array(pixels).T
                yield label, ys, xs, (xs.min(), ys.min(), min(width, xs.max() + 1), min(height, ys.max() + 1))


def box_iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area_a + area_b - overlap
    return overlap / union if union else 0


def scene_evaluation(kind, reference, mask_data, image, classes, objects, rows):
    import numpy as np
    from PIL import Image

    if not isinstance(reference, dict):
        raise InputError("图片参考标注格式有误。")
    if kind != "detection":
        try:
            with Image.open(io.BytesIO(binary_input(mask_data, "参考掩膜")), formats=("PNG",)) as mask:
                if mask.mode != "L" or mask.size != image.size:
                    raise InputError("参考掩膜需要与图片同尺寸，使用灰度 PNG 存储类别或实例编号。")
                truth = np.asarray(mask.resize((classes.shape[1], classes.shape[0]), Image.Resampling.NEAREST))
        except (OSError, Image.DecompressionBombError) as error:
            raise InputError("无法读取参考掩膜。") from error
        if kind == "semantic":
            if truth.max() > 3:
                raise InputError("语义参考掩膜的像素类别必须为 0–3。")
            ious = []
            for label in range(1, 4):
                actual, predicted = truth == label, classes == label
                if union := (actual | predicted).sum():
                    ious.append(float((actual & predicted).sum() / union))
            metrics = [metric("像素准确率", float((truth == classes).mean()), "percent")]
            if ious:
                metrics.append(metric("前景平均交并比", sum(ious) / len(ious), "percent"))
            return metrics

    targets = reference.get("objects")
    if not isinstance(targets, list) or len(targets) > 32:
        raise InputError("参考物体列表格式有误。")
    matches = []
    for j, target in enumerate(targets):
        if not isinstance(target, dict) or type(target.get("class_id")) is not int or not 1 <= target["class_id"] <= 3:
            raise InputError("参考物体需要有效的类别编号。")
        if kind == "detection":
            box = target.get("bbox_xyxy")
            if (not isinstance(box, list) or len(box) != 4 or not all(type(v) is int for v in box)
                    or not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
                raise InputError("参考检测框需要包含图像内的四个整数坐标。")
        else:
            instance = target.get("instance")
            if type(instance) is not int or not 1 <= instance <= 32:
                raise InputError("参考实例编号必须为 1–32。")
            actual = truth == instance
        for i, (label, ys, xs, _) in enumerate(objects):
            if label != target["class_id"]:
                continue
            if kind == "detection":
                score = box_iou(box, rows[i][3:])
            else:
                intersection = actual[ys, xs].sum()
                score = float(intersection / (actual.sum() + len(xs) - intersection))
            if score >= .5:
                matches.append((score, i, j))
    seen_predictions, seen_targets = set(), set()
    for _, i, j in sorted(matches, reverse=True):
        if i not in seen_predictions and j not in seen_targets:
            seen_predictions.add(i)
            seen_targets.add(j)
    count = len(seen_targets)
    return [metric("参考物体", len(targets)), metric("正确匹配（IoU ≥ 0.5）", count),
            metric("漏检", len(targets) - count), metric("多检", len(objects) - count)]


def scene_predict(kind):
    def infer(network, payload):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from classic_vision.train import CLASSES, PALETTE, SIZE, image_tensor

        image = load_image(payload)
        resized = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        with torch.inference_mode():
            softmax = network(image_tensor(resized)[None]).softmax(dim=1)[0]
            classes = softmax.argmax(dim=0).numpy()
        objects = []
        if kind == "semantic":
            preview = Image.fromarray(np.array(PALETTE, dtype=np.uint8)[classes])
            rows = [[name, float((classes == label).mean())] for label, name in enumerate(CLASSES)]
            columns = ("class_name", "pixel_fraction")
        else:
            preview = resized.copy()
            objects = sorted(connected_objects(classes), key=lambda item: (item[3][1], item[3][0]))
            rows = []
            if kind == "instance":
                colors = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
                rng = Random(76)
            for index, (label, ys, xs, box) in enumerate(objects, 1):
                score = float(softmax[label, ys, xs].mean())
                x1, y1, x2, y2 = box
                rows.append([index, CLASSES[label], score, round(x1 * image.width / SIZE), round(y1 * image.height / SIZE),
                             round(x2 * image.width / SIZE), round(y2 * image.height / SIZE)])
                if kind == "instance":
                    colors[ys, xs] = tuple(rng.randint(80, 255) for _ in range(3))
                else:
                    pen = ImageDraw.Draw(preview)
                    pen.rectangle((x1, y1, x2 - 1, y2 - 1), outline="white", width=1)
                    pen.text((x1, max(0, y1 - 9)), str(index), fill="white")
            if kind == "instance":
                preview = Image.fromarray(colors)
            columns = ("instance", "class_name", "probability", "x_min", "y_min", "x_max", "y_max")
        result = {"columns": columns, "rows": rows, "preview": {"type": "image/png", "data": encode(png(preview))}}
        if payload.get("reference") is not None:
            metrics = scene_evaluation(kind, payload["reference"], payload.get("reference_mask"), image, classes, objects, rows)
            result["evaluation"] = {"count": 1, "metrics": metrics}
        return result
    return infer


def rotate_image(image, degrees, fill, resampling):
    if not degrees:
        return image
    rotated = image.rotate(degrees, resample=resampling, expand=True, fillcolor=fill)
    return rotated.resize(image.size, resampling)


def occluded_pixels(mask, percent):
    import numpy as np

    ys, xs = np.nonzero(mask)
    count = min(max(0, len(ys) - 1), round(len(ys) * percent / 100))
    if not count:
        return ys[:0], xs[:0]
    center = (ys.min() + ys.max()) / 2
    selected = np.argsort(np.abs(ys - center), kind="stable")[:count]
    return ys[selected], xs[selected]


def digit_variation(image, options):
    import numpy as np
    from PIL import Image

    size = options["size_percent"]
    if size != 100:
        glyph = image.crop(image.getbbox())
        factor = min(size / 100, min(image.width / glyph.width, image.height / glyph.height))
        glyph = glyph.resize(tuple(max(1, round(side * factor)) for side in glyph.size), Image.Resampling.BILINEAR)
        image = Image.new("L", image.size)
        image.paste(glyph, ((image.width - glyph.width) // 2, (image.height - glyph.height) // 2))
    image = rotate_image(image, options["rotation_degrees"], 0, Image.Resampling.BILINEAR)
    if options["occlusion_percent"]:
        pixels = np.array(image)
        ys, xs = occluded_pixels(pixels > 12, options["occlusion_percent"])
        pixels[ys, xs] = 0
        image = Image.fromarray(pixels)
    return image


def scene_variation(image, instances, objects, options):
    import numpy as np
    from PIL import Image

    angle = options["rotation_degrees"]
    background = image.getpixel((0, 0))
    image = rotate_image(image, angle, background, Image.Resampling.BILINEAR)
    instances = rotate_image(instances, angle, 0, Image.Resampling.NEAREST)
    pixels = np.array(image)
    instance_ids = np.array(instances)
    class_ids = np.zeros_like(instance_ids)
    visible_objects = []
    for item in objects:
        instance = item["instance"]
        covered_y, covered_x = occluded_pixels(instance_ids == instance, options["occlusion_percent"])
        instance_ids[covered_y, covered_x] = 0
        pixels[covered_y, covered_x] = background
        ys, xs = np.nonzero(instance_ids == instance)
        if not len(ys):
            continue
        class_ids[ys, xs] = item["class_id"]
        visible_objects.append({**item, "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                                "visible_pixels": len(ys), "occluded_pixels": len(covered_y)})
    return Image.fromarray(pixels), Image.fromarray(class_ids), Image.fromarray(instance_ids), visible_objects


def image_dataset(kind):
    def generate(module, options):
        rng = Random(options["seed"])
        samples, files = [], []
        for i in range(options["samples"]):
            name = f"sample-{i + 1:02}.png"
            target = {}
            reference_mask = None
            if kind == "digit":
                label = (rng.randrange(10) if options["digit"] == -1 else options["digit"])
                image = digit_variation(module.drawn_digit(label, rng), options)
                target["label"] = label
            else:
                image, _, instances, objects = module.draw_scene(rng, options["objects"], options["size_percent"])
                image, semantic, instances, objects = scene_variation(image, instances, objects, options)
                target["objects"] = objects
                if kind in ("semantic", "instance"):
                    mask = semantic if kind == "semantic" else instances
                    target["mask"] = f"mask-{i + 1:02}.png"
                    mask_bytes = png(mask)
                    files.append((target["mask"], mask_bytes))
                    reference_mask = encode(mask_bytes)
            payload = png(image)
            files.append((name, payload))
            samples.append({"name": name, "type": "image/png", "data": encode(payload), "target": target,
                            **({"reference_mask": reference_mask} if reference_mask else {})})
        return {"samples": samples, "archive": dataset_zip(files, [{"image": s["name"], **s["target"]} for s in samples])}
    return generate
