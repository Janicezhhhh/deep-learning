import io
from collections import deque
from random import Random

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
    return {"columns": ("predicted_label", "probability"), "rows": rows}


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
        return {"columns": columns, "rows": rows, "preview": {"type": "image/png", "data": encode(png(preview))}}
    return infer


def image_dataset(kind):
    def generate(module, options):
        rng = Random(options["seed"])
        samples, files = [], []
        for i in range(options["samples"]):
            name = f"sample-{i + 1:02}.png"
            target = {}
            if kind == "digit":
                label = (rng.randrange(10) if options["digit"] == -1 else options["digit"])
                image = module.drawn_digit(label, rng)
                target["label"] = label
            else:
                image, semantic, instances, objects = module.draw_scene(rng, options["objects"])
                target["objects"] = objects
                if kind in ("semantic", "instance"):
                    mask = semantic if kind == "semantic" else instances
                    target["mask"] = f"mask-{i + 1:02}.png"
                    files.append((target["mask"], png(mask)))
            payload = png(image)
            files.append((name, payload))
            samples.append({"name": name, "type": "image/png", "data": encode(payload), "target": target})
        return {"samples": samples, "archive": dataset_zip(files, [{"image": s["name"], **s["target"]} for s in samples])}
    return generate
