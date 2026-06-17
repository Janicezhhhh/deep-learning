import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 500, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 3), generator=generator)
    weights = torch.tensor([2.0, -1.5, 0.5])
    y = x @ weights + 0.7 * torch.randn(n_samples, generator=generator)
    y = y.unsqueeze(1)
    return x, y


class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def plot_regression(y_true, y_pred):
    y_true = y_true.cpu().numpy().squeeze(1)
    y_pred = y_pred.cpu().numpy().squeeze(1)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, s=40)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.title("Regression True vs Predicted")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_plot.png")
    plt.close()


def save_io_text(path, x, y_true, y_pred):
    x = x.cpu()
    y_true = y_true.cpu().squeeze(1)
    y_pred = y_pred.cpu().squeeze(1)
    with open(path, "w") as f:
        f.write("x1 x2 x3 true_value predicted_value\n")
        for i in range(len(x)):
            f.write(f"{x[i,0].item():.4f} {x[i,1].item():.4f} {x[i,2].item():.4f} {y_true[i].item():.4f} {y_pred[i].item():.4f}\n")


def main():
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_data()
    split = int(len(x) * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = RegressionMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(150):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d} | Train loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(x_test.to(device))
        test_loss = criterion(test_preds, y_test.to(device)).item()

    plot_regression(y_test, test_preds)
    save_io_text("regression_io.txt", x_test, y_test, test_preds)

    save_path = Path("regression_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete. Test MSE: {test_loss:.4f}")
    print(f"Model saved to: {save_path.resolve()}")
    print(f"Visualization saved to: {Path('regression_plot.png').resolve()}")
    print(f"IO data saved to: {Path('regression_io.txt').resolve()}")


if __name__ == "__main__":
    main()
