"""Optional local Transformer checkpoints and their upstream model cards."""

from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent.parent / ".workbench_models"
REPOSITORIES = {
    "sentiment": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "entities": "dslim/distilbert-NER",
    "translation": "Helsinki-NLP/opus-mt-en-zh",
    "summary": "google/flan-t5-small",
    "question": "distilbert/distilbert-base-cased-distilled-squad",
    "generation": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "keywords": "superb/wav2vec2-base-superb-ks",
    "transcription": "openai/whisper-tiny.en",
    "speech": "facebook/mms-tts-eng",
}


def model_directory(key):
    return MODEL_ROOT / key


def available(key):
    path = model_directory(key)
    return ((path / "config.json").is_file() and (path / ".ready").is_file()
            and ((path / "pytorch_model.bin").is_file() or any(path.glob("*.safetensors"))))
