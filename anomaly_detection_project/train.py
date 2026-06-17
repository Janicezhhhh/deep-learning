import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_normal: int = 400, n_anomaly: int = 50, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    normal = torch.randn((n_normal, 2), generator=generator) * 0.8
    anomaly = (torch.rand((n_anomaly, 2), generator=generator) - 0.5) * 10
    x = torch.vstack([normal, anomaly])
    labels = torch.cat([torch.zeros(n_normal), torch.ones(n_anomaly)])
    return x, labels


class AnomalyAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def compute_anomaly_score(x, decoded):
    mse = torch.mean((x - decoded) ** 2, dim=1)
    score = (mse - mse.min()) / (mse.max() - mse.min())
    return score


def plot_anomaly(x, labels, scores):
    x = x.cpu()
    labels = labels.cpu()
    scores = scores.cpu()
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=scores, cmap="RdYlGn_r", alpha=0.7, s=50)
    plt.colorbar(scatter, label="Anomaly Score")
    plt.title("Anomaly Detection Input/Output Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("anomaly_plot.png")
    plt.close()


def save_io_text(path, x, labels, scores):
    x = x.cpu()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()
    with open(path, "w") as f:
        f.write("x1 x2 true_label anomaly_score prediction\n")
        for i in range(len(x)):
            pred = 1 if scores[i] > 0.5 else 0
            f.write(f"{x[i,0].item():.4f} {x[i,1].item():.4f} {int(labels[i])} {scores[i]:.4f} {pred}\n")


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, labels = make_data()
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AnomalyAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            z, decoded = model(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        z, decoded = model(x.to(device))
        scores = compute_anomaly_score(x.to(device), decoded)

    plot_anomaly(x, labels, scores)
    save_io_text("anomaly_results.txt", x, labels, scores)

    save_path = Path("anomaly_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete.")
    print(f"Model saved to: {save_path.resolve()}")
    print(f"Visualization saved to: {Path('anomaly_plot.png').resolve()}")
    print(f"Results saved to: {Path('anomaly_results.txt').resolve()}")


if __name__ == "__main__":
    main()
