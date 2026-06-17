import torch
from pathlib import Path
import matplotlib.pyplot as plt


def make_data(n_samples: int = 300, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[2.0, 2.0], [6.0, 6.0], [-4.0, 1.0]])
    points = []
    labels = []
    for i, center in enumerate(centers):
        cluster = center + 0.7 * torch.randn((n_samples // 3, 2), generator=generator)
        points.append(cluster)
        labels.append(torch.full((n_samples // 3,), i, dtype=torch.long))
    x = torch.vstack(points)
    y = torch.cat(labels)
    return x, y, centers


def kmeans(x, k=3, n_iters=20):
    centers = x[torch.randperm(len(x))[:k]].clone()
    for _ in range(n_iters):
        dist = torch.cdist(x, centers)
        labels = torch.argmin(dist, dim=1)
        new_centers = torch.stack([x[labels == i].mean(dim=0) for i in range(k)])
        centers = new_centers
    return centers, labels


def plot_clusters(x, labels, centers, sample_inputs=None, sample_labels=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=150, label="Cluster centers")

    if sample_inputs is not None and sample_labels is not None:
        for i, sample in enumerate(sample_inputs):
            plt.scatter(sample[0], sample[1], c="red", marker="*", s=200)
            plt.text(sample[0] + 0.1, sample[1] + 0.1, f"sample {i}: {sample_labels[i].item()}", color="red")

    plt.title("Clustering Input/Output Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("cluster_plot.png")
    plt.close()


def main():
    torch.manual_seed(1)
    x, y, true_centers = make_data()
    centers, labels = kmeans(x, k=3, n_iters=15)

    sample_inputs = torch.tensor([[1.0, 1.0], [5.0, 5.0], [-4.0, 0.0]])
    sample_dist = torch.cdist(sample_inputs, centers)
    sample_labels = torch.argmin(sample_dist, dim=1)

    plot_clusters(x, labels, centers, sample_inputs=sample_inputs, sample_labels=sample_labels)

    save_path = Path("cluster_centers.pth")
    torch.save(centers, save_path)

    print("Sample inputs:", sample_inputs.tolist())
    print("Predicted cluster labels:", sample_labels.tolist())
    print(f"Cluster centers saved to: {save_path.resolve()}")
    print(f"Visualization saved to: {Path('cluster_plot.png').resolve()}")


if __name__ == "__main__":
    main()
