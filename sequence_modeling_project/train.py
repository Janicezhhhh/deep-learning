import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_sequence_data(length: int = 300, seq_len: int = 10, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * 3.14159, length)
    data = torch.sin(t) + 0.1 * torch.randn(length, generator=generator)
    x, y = [], []
    for i in range(length - seq_len - 1):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.stack(x), torch.stack(y).unsqueeze(1)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 16, seq_len: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(-1))
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


def plot_sequence(y_test, y_pred):
    y_test = y_test.cpu().numpy().squeeze(1)
    y_pred = y_pred.cpu().detach().numpy().squeeze(1)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], label="True", marker="o", markersize=4, alpha=0.7)
    plt.plot(y_pred[:100], label="Predicted", marker="s", markersize=4, alpha=0.7)
    plt.title("Sequence Modeling: True vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("sequence_plot.png")
    plt.close()


def save_io_text(path, y_test, y_pred):
    y_test = y_test.cpu().numpy().squeeze(1)
    y_pred = y_pred.cpu().detach().numpy().squeeze(1)
    with open(path, "w") as f:
        f.write("true_value predicted_value\n")
        for i in range(min(100, len(y_test))):
            f.write(f"{y_test[i]:.4f} {y_pred[i]:.4f}\n")


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_sequence_data()
    split = int(len(x) * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
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

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test.to(device))
        test_loss = criterion(y_pred, y_test.to(device)).item()

    plot_sequence(y_test, y_pred)
    save_io_text("sequence_io.txt", y_test, y_pred)

    save_path = Path("sequence_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete. Test MSE: {test_loss:.4f}")
    print(f"Model saved to: {save_path.resolve()}")
    print(f"Visualization saved to: {Path('sequence_plot.png').resolve()}")
    print(f"IO data saved to: {Path('sequence_io.txt').resolve()}")


if __name__ == "__main__":
    main()
