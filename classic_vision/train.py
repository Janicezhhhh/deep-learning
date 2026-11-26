"""Train the digit classifier and the geometric-scene segmentation network."""

import argparse
import gzip
from pathlib import Path
from random import Random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SIZE = 96
CLASSES = ("背景", "圆形", "方形", "三角形")
PALETTE = ((0, 0, 0), (255, 72, 94), (61, 227, 149), (89, 155, 255))
DIGIT_STROKES = (
    (((17, 5), (9, 5), (6, 10), (5, 18), (8, 23), (16, 24), (21, 19), (22, 10), (17, 5)),),
    (((9, 10), (15, 5), (15, 23)),),
    (((6, 9), (10, 5), (17, 5), (21, 9), (20, 13), (6, 23), (22, 23)),),
    (((6, 7), (14, 4), (21, 8), (18, 13), (13, 14), (20, 16), (21, 21), (14, 24), (6, 22)),),
    (((17, 4), (6, 17), (22, 17)), ((18, 7), (17, 24))),
    (((21, 5), (8, 5), (6, 13), (17, 12), (21, 16), (20, 21), (15, 24), (6, 22)),),
    (((20, 5), (13, 5), (8, 12), (6, 19), (10, 24), (17, 24), (21, 19), (19, 15), (13, 14), (8, 17)),),
    (((6, 6), (22, 5), (13, 24)),),
    (((13, 13), (6, 8), (9, 4), (17, 4), (21, 8), (17, 13), (8, 17), (7, 21), (12, 24), (19, 23), (22, 19), (13, 13)),),
    (((20, 13), (15, 15), (8, 14), (6, 10), (9, 5), (17, 4), (21, 9), (20, 19), (15, 24), (9, 24)),),
)


class DigitCNN(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(24 * 7 * 7, 48), nn.ReLU(), nn.Linear(48, 10))


class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
                                     nn.Upsample(scale_factor=2, mode="nearest"))
        self.head = nn.Sequential(nn.Conv2d(40, 20, 3, padding=1), nn.ReLU(), nn.Conv2d(20, 4, 1))

    def forward(self, x):
        details = self.detail(x)
        return self.head(torch.cat((details, self.context(details)), dim=1))


def drawn_digit(label, rng):
    image = Image.new("L", (28, 28))
    pen = ImageDraw.Draw(image)
    for stroke in DIGIT_STROKES[label]:
        points = [(x + rng.randint(-1, 1), y + rng.randint(-1, 1)) for x, y in stroke]
        pen.line(points, fill=255, width=rng.randint(2, 4), joint="curve")
    return image.rotate(rng.randint(-14, 14), resample=Image.Resampling.BILINEAR,
                        translate=(rng.randint(-2, 2), rng.randint(-2, 2)))


def draw_scene(rng, objects, size_percent=100):
    level = rng.randint(8, 35)
    image = Image.new("RGB", (SIZE, SIZE), (level, level, level + 5))
    semantic = Image.new("L", (SIZE, SIZE))
    instances = Image.new("L", (SIZE, SIZE))
    image_pen, class_pen, instance_pen = map(ImageDraw.Draw, (image, semantic, instances))
    annotations = []
    for index, cell in enumerate(rng.sample(range(9), objects), 1):
        label = rng.randrange(1, 4)
        left, top = rng.randint(2, 7), rng.randint(2, 7)
        x = (cell % 3) * 32 + left
        y = (cell // 3) * 32 + top
        side = min(round(rng.randint(15, 23) * size_percent / 100), 30 - max(left, top))
        box = (x, y, x + side, y + side)
        points = [(x + side // 2, y), (x, y + side), (x + side, y + side)] if label == 3 else box
        method = ("ellipse", "rectangle", "polygon")[label - 1]
        color = tuple(max(0, min(255, c + rng.randint(-30, 15))) for c in PALETTE[label])
        for pen, fill in ((image_pen, color), (class_pen, label), (instance_pen, index)):
            getattr(pen, method)(points, fill=fill)
        annotations.append({"instance": index, "class_id": label, "class_name": CLASSES[label],
                            "bbox_xyxy": [x, y, x + side + 1, y + side + 1]})
    return image, semantic, instances, annotations


def image_tensor(image):
    values = torch.from_numpy(np.array(image)).float().div_(255)
    return values.unsqueeze(0) if values.ndim == 2 else values.permute(2, 0, 1)


def mnist(directory, prefix):
    images = gzip.decompress((directory / f"{prefix}-images-idx3-ubyte.gz").read_bytes())
    labels = gzip.decompress((directory / f"{prefix}-labels-idx1-ubyte.gz").read_bytes())
    digits = torch.frombuffer(bytearray(images[16:]), dtype=torch.uint8).reshape(-1, 1, 28, 28)
    targets = torch.frombuffer(bytearray(labels[8:]), dtype=torch.uint8).long()
    return digits.float() / 255, targets


def train_model(model, x, y, val_x, val_y, criterion, epochs, batch_size, filename):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()
        print(f"{filename}: epoch {epoch + 1}/{epochs}", flush=True)
    with torch.inference_mode():
        model.eval()
        predicted = torch.cat([model(batch).argmax(dim=1) for batch in val_x.split(100)])
        accuracy = (predicted == val_y).float().mean().item()
        print(f"{filename}: holdout accuracy {accuracy:.4f}", flush=True)
    torch.save(model.state_dict(), Path(__file__).with_name(filename))


def train_digits(path):
    rng = Random(42)
    digits, labels = mnist(path, "train")
    selected = torch.randperm(len(labels))[:24000]
    synthetic = torch.arange(2400) % 10
    generated = torch.stack([image_tensor(drawn_digit(label, rng)) for label in synthetic.tolist()])
    val_x, val_y = mnist(path, "t10k")
    train_model(DigitCNN(), torch.cat((digits[selected], generated)), torch.cat((labels[selected], synthetic)),
                val_x, val_y, nn.CrossEntropyLoss(), 3, 128, "digit_cnn.pth")


def scene_tensors(seed, count):
    rng = Random(seed)
    pairs = [draw_scene(rng, rng.randint(1, 5))[:2] for _ in range(count)]
    return (torch.stack([image_tensor(im) for im, _ in pairs]),
            torch.stack([torch.from_numpy(np.array(mask)) for _, mask in pairs]).long())


def train_scenes():
    x, y = scene_tensors(42, 200)
    test_x, test_y = scene_tensors(91, 32)
    train_model(ShapeNet(), x, y, test_x, test_y, nn.CrossEntropyLoss(), 5, 12, "shape_net.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", type=Path, help="Directory containing the four compressed MNIST IDX files")
    parser.add_argument("--shapes", action="store_true", help="Train the geometric-scene model")
    args = parser.parse_args()
    if not args.mnist and not args.shapes:
        parser.error("Provide --mnist or --shapes")
    torch.manual_seed(42)
    torch.set_num_threads(3)
    if args.mnist:
        train_digits(args.mnist)
    if args.shapes:
        train_scenes()
