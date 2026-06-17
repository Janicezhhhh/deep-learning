import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path


def make_interaction_matrix(n_users: int = 50, n_items: int = 30, density: float = 0.3, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.rand((n_users, n_items), generator=generator)
    mask = torch.rand((n_users, n_items), generator=generator) < density
    interaction = torch.where(mask, torch.round(matrix * 4) + 1, torch.zeros_like(matrix))
    return interaction, mask


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int = 10):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

    def forward(self, user_idx, item_idx):
        user_vec = self.user_emb(user_idx)
        item_vec = self.item_emb(item_idx)
        pred = (user_vec * item_vec).sum(dim=1) + self.user_bias(user_idx).squeeze(1) + self.item_bias(item_idx).squeeze(1)
        return torch.clamp(pred, min=1, max=5)


def plot_recommendations(pred_matrix):
    pred_matrix = pred_matrix.cpu().numpy()
    plt.figure(figsize=(10, 6))
    im = plt.imshow(pred_matrix[:20, :15], cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, label="Predicted Rating")
    plt.title("User-Item Prediction Matrix")
    plt.xlabel("Item")
    plt.ylabel("User")
    plt.tight_layout()
    plt.savefig("recommendation_heatmap.png")
    plt.close()


def save_io_text(path, interaction, mask, pred_matrix):
    with open(path, "w") as f:
        f.write("Sample predictions for observed interactions:\n")
        f.write("user_id item_id true_rating predicted_rating\n")
        count = 0
        for u in range(min(5, interaction.shape[0])):
            for i in range(min(5, interaction.shape[1])):
                if mask[u, i]:
                    f.write(f"{u} {i} {interaction[u, i].item():.0f} {pred_matrix[u, i].item():.2f}\n")
                    count += 1
                    if count >= 20:
                        break
            if count >= 20:
                break


def main():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_users, n_items = 50, 30
    interaction, mask = make_interaction_matrix(n_users, n_items)

    model = MatrixFactorization(n_users, n_items, k=10).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    user_indices, item_indices = torch.where(mask)
    user_indices = user_indices.to(device)
    item_indices = item_indices.to(device)
    ratings = interaction[mask].to(device)

    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        preds = model(user_indices, item_indices)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:03d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        all_user_idx = torch.arange(n_users, device=device).repeat_interleave(n_items)
        all_item_idx = torch.arange(n_items, device=device).repeat(n_users)
        pred_matrix = model(all_user_idx, all_item_idx).reshape(n_users, n_items)

    plot_recommendations(pred_matrix)
    save_io_text("recommendation_io.txt", interaction, mask, pred_matrix)

    save_path = Path("recommendation_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Training complete.")
    print(f"Model saved to: {save_path.resolve()}")
    print(f"Heatmap saved to: {Path('recommendation_heatmap.png').resolve()}")
    print(f"IO data saved to: {Path('recommendation_io.txt').resolve()}")


if __name__ == "__main__":
    main()
