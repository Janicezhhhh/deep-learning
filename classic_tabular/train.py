"""Train reproducible demo models for the workbench's numeric tasks."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MulticlassMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 3))


class MultilabelMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(3, 24), nn.ReLU(), nn.Linear(24, 3))


class ForecastLSTM(nn.Module):
    def __init__(self, features=1, steps=3):
        super().__init__()
        self.features = features
        self.lstm = nn.LSTM(features, 24, batch_first=True)
        self.head = nn.Linear(24, steps)

    def forward(self, x):
        sequence = x.reshape(len(x), 10, self.features)
        hidden, _ = self.lstm(sequence)
        return self.head(hidden[:, -1])


def multiclass_data(count, seed, *, class_zero_percent=33, noise_percent=100, distance_percent=100):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2., -1.], [2., -1.], [0., 2.5]]) * (distance_percent / 100)
    if class_zero_percent == 33:
        labels = torch.randint(3, (count,), generator=generator)
    else:
        remaining = (100 - class_zero_percent) / 2
        weights = torch.tensor([class_zero_percent, remaining, remaining])
        labels = torch.multinomial(weights, count, replacement=True, generator=generator)
    x = centers[labels] + (.62 * noise_percent / 100) * torch.randn((count, 2), generator=generator)
    return x, labels


def multilabel_data(count, seed, *, noise_percent=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((count, 3), generator=generator)
    labels = torch.stack((x[:, 0] > 0, x[:, 1] > x[:, 2], x.sum(dim=1) > 0), dim=1)
    if noise_percent:
        x = x + (noise_percent / 100) * torch.randn(x.shape, generator=generator)
    return x, labels.float()


def forecast_data(count, seed, features=1, steps=3, *, trend_percent=0, period_percent=100, jump_percent=0):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.rand((count, 1), generator=generator) * (2 * torch.pi)
    pace = .16 + .08 * torch.rand((count, 1), generator=generator)
    ticks = torch.arange(10 + steps)
    time = phase + pace * (100 / period_percent) * ticks
    auxiliary = torch.cos(time)
    noise = .035 * torch.randn(time.shape, generator=generator)
    target = torch.sin(time) + noise
    target = target + (trend_percent / 1000) * ticks + (jump_percent / 100) * (ticks >= 7)
    x = target[:, :10]
    if features == 2:
        target = target + .35 * auxiliary
        x = torch.stack((target[:, :10], auxiliary[:, :10]), dim=-1).flatten(1)
    return x, target[:, 10:]


def temporal_anomaly_data(count, seed, **controls):
    rows, values = forecast_data(count, seed, steps=1, **controls)
    generator = torch.Generator().manual_seed(seed ^ 42)
    labels = torch.rand((count, 1), generator=generator) < .2
    direction = torch.where(torch.rand((count, 1), generator=generator) < .5, -1, 1)
    observed = values + labels * direction * 1.5
    return torch.cat((rows, observed), dim=1), labels.float()


def train(model, source, criterion, filename):
    x, y = source(2000, 42)
    x_test, y_test = source(400, 91)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.007)
    for _ in range(22):
        for inputs, targets in loader:
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()
    with torch.inference_mode():
        loss = criterion(model.eval()(x_test), y_test).item()
    torch.save(model.state_dict(), Path(__file__).with_name(filename))
    print(f"{filename}: holdout loss {loss:.4f}", flush=True)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    train(MulticlassMLP(), multiclass_data, nn.CrossEntropyLoss(), "multiclass.pth")
    train(MultilabelMLP(), multilabel_data, nn.BCEWithLogitsLoss(), "multilabel.pth")
    train(ForecastLSTM(), forecast_data, nn.MSELoss(), "multistep.pth")
    train(ForecastLSTM(2, 1), lambda n, seed: forecast_data(n, seed, 2, 1), nn.MSELoss(), "multivariate.pth")


if __name__ == "__main__":
    main()
