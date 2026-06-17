import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 400, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 2), generator=generator)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).float().unsqueeze(1)
    return x, y


class SimpleMLP(nn.Module):
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


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_data()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
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

    save_path = Path("model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete. Accuracy: {accuracy:.2%}")
    print(f"Model saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
