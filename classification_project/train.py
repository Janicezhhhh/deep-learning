import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 400, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 2), generator=generator)
    y = ((x[:, 0] * x[:, 1]) > 0).float().unsqueeze(1)
    return x, y


class ClassificationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def plot_classification(x, y, preds):
    x = x.cpu()
    y = y.cpu().squeeze(1)
    preds = preds.cpu().squeeze(1)
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=40, edgecolors="k", linewidths=0.3)
    wrong = preds != y
    if wrong.any():
        plt.scatter(x[wrong, 0], x[wrong, 1], facecolors="none", edgecolors="black", s=120, linewidths=1.5, label="misclassified")
    plt.title("Classification Input/Output Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("classification_plot.png")
    plt.close()


def save_io_text(path, x, y, preds):
    x = x.cpu()
    y = y.cpu().squeeze(1)
    preds = preds.cpu().squeeze(1)
    with open(path, "w") as f:
        f.write("x1 x2 true_label pred_label\n")
        for i in range(len(x)):
            f.write(f"{x[i,0].item():.4f} {x[i,1].item():.4f} {int(y[i].item())} {int(preds[i].item())}\n")


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_data()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ClassificationMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(150):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (preds == y.to(device)).float().mean().item()

    plot_classification(x, y, preds)
    save_io_text("classification_io.txt", x, y, preds)

    save_path = Path("classification_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete. Accuracy: {accuracy:.2%}")
    print(f"Model saved to: {save_path.resolve()}")
    print(f"Visualization saved to: {Path('classification_plot.png').resolve()}")
    print(f"IO data saved to: {Path('classification_io.txt').resolve()}")


if __name__ == "__main__":
    main()
