"""Download selected open checkpoints for offline inference."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from .model_assets import REPOSITORIES, available, model_directory


def download(key):
    from huggingface_hub import HfApi, snapshot_download

    path = model_directory(key)
    if available(key):
        return
    repository = REPOSITORIES[key]
    info = HfApi().model_info(repository, token=False)
    names = [file.rfilename for file in info.siblings if "/" not in file.rfilename]
    files = [name for name in names if name.endswith((".json", ".txt", ".model", ".spm"))]
    weights = [name for name in names if name.endswith(".safetensors")]
    if not weights and "pytorch_model.bin" in names:
        weights = ["pytorch_model.bin"]
    if not weights:
        raise RuntimeError(f"No supported PyTorch checkpoint in {repository}")
    print(f"Downloading {key} from {repository}", flush=True)
    snapshot_download(repository, revision=info.sha, local_dir=path, allow_patterns=files + weights,
                      token=False, max_workers=3)
    (path / ".ready").write_text(info.sha + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Download weights for local text and audio tasks")
    parser.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(REPOSITORIES))
    args = parser.parse_args()
    keys = tuple(dict.fromkeys(args.models or REPOSITORIES))
    unsupported = set(keys) - REPOSITORIES.keys()
    if unsupported:
        parser.error("Unknown models: " + ", ".join(sorted(unsupported)))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "40")
    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [(key, pool.submit(download, key)) for key in keys]
        for key, job in jobs:
            try:
                job.result()
                print(f"{key}: ready", flush=True)
            except Exception as error:
                failures.append(key)
                print(f"{key}: {type(error).__name__}: {error}", flush=True)
    if failures:
        parser.exit(1, "Download again for: " + " ".join(failures) + "\n")


if __name__ == "__main__":
    main()
