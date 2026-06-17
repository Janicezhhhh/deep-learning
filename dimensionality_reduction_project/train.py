import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def make_data(n_samples: int = 500, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 4), generator=generator) * torch.tensor([2.0, 1.0, 3.0, 0.5])
    return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def plot_latent(z):
    z = z.cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.6, s=40)
    plt.title("Latent Space Visualization")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("autoencoder_latent_plot.png")
    plt.close()


def save_io_text(path, input_data, encoded, decoded):
    input_data = input_data.cpu()
    encoded = encoded.cpu()
    decoded = decoded.cpu()
    with open(path, "w") as f:
        f.write("x1 x2 x3 x4 encoded1 encoded2 decoded1 decoded2 decoded3 decoded4\n")
        for i in range(len(input_data)):
            row = [input_data[i, j].item() for j in range(4)] + [encoded[i, j].item() for j in range(2)] + [decoded[i, j].item() for j in range(4)]
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")


def main():
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = make_data()
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(120):
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

        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        z_all, decoded_all = model(x.to(device))

    plot_latent(z_all)
    save_io_text("autoencoder_io.txt", x, z_all, decoded_all)

    save_path = Path("autoencoder_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Latent visualization saved to: {Path('autoencoder_latent_plot.png').resolve()}")
    print(f"IO data saved to: {Path('autoencoder_io.txt').resolve()}")
    print(f"Model saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
