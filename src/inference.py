"""Inference pipeline shared by the Gradio app and the evaluation code.

    raw review -> clean_text -> selected model -> probabilities
               -> (sentiment, confidence) -> CS flagging logic
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src import config
from src.data_utils import clean_text, get_device, pad_batch
from src.flagging_logic import FlagResult, flag_review


@dataclass(frozen=True)
class Prediction:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    model_name: str
    prediction: Prediction
    flag: FlagResult


class SentimentPredictor(ABC):
    """Common interface: raw texts in, class probabilities out."""

    name: str

    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if device else get_device()

    @abstractmethod
    def _predict_proba_clean(self, texts: Sequence[str]) -> np.ndarray:
        """Probabilities of shape (N, NUM_CLASSES) for already-cleaned texts."""

    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        batches = [
            self._predict_proba_clean(cleaned[i : i + batch_size])
            for i in range(0, len(cleaned), batch_size)
        ]
        return np.concatenate(batches) if batches else np.empty((0, config.NUM_CLASSES))

    def predict(self, text: str) -> Prediction:
        probs = self.predict_proba([text])[0]
        idx = int(probs.argmax())
        return Prediction(
            sentiment=config.ID2LABEL[idx],
            confidence=float(probs[idx]),
            probabilities={config.ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        )


class BiLSTMPredictor(SentimentPredictor):
    name = "BiLSTM"

    def __init__(
        self,
        checkpoint_path: Path = config.BILSTM_CHECKPOINT_PATH,
        device: torch.device | str | None = None,
    ):
        super().__init__(device)
        from src.model_bilstm import load_checkpoint

        self.model, self.vocab, checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.max_len = checkpoint.get("max_len", config.BILSTM_MAX_LEN)

    @torch.no_grad()
    def _predict_proba_clean(self, texts: Sequence[str]) -> np.ndarray:
        sequences = [torch.tensor(self.vocab.encode(t, self.max_len)) for t in texts]
        input_ids, lengths = pad_batch(sequences, self.vocab.pad_idx)
        logits = self.model(input_ids.to(self.device), lengths)
        return torch.softmax(logits, dim=-1).cpu().numpy()


class DistilBERTPredictor(SentimentPredictor):
    name = "DistilBERT"

    def __init__(
        self,
        model_dir: Path = config.DISTILBERT_DIR,
        device: torch.device | str | None = None,
    ):
        super().__init__(device)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (Path(model_dir) / "config.json").exists():
            raise FileNotFoundError(
                f"DistilBERT model not found in {model_dir}. "
                "Fine-tune it with notebooks/02_finetune_distilbert.ipynb."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _predict_proba_clean(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()


MODEL_REGISTRY: dict[str, type[SentimentPredictor]] = {
    BiLSTMPredictor.name: BiLSTMPredictor,
    DistilBERTPredictor.name: DistilBERTPredictor,
}
AVAILABLE_MODELS = tuple(MODEL_REGISTRY)


@lru_cache(maxsize=None)
def get_predictor(model_name: str) -> SentimentPredictor:
    """Load a model once and reuse it for every subsequent request."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {AVAILABLE_MODELS}.")
    return MODEL_REGISTRY[model_name]()


def analyze_review(text: str, model_name: str = "DistilBERT") -> AnalysisResult:
    """Full pipeline for one review: prediction + CS flag."""
    prediction = get_predictor(model_name).predict(text)
    flag = flag_review(prediction.sentiment, prediction.confidence, text)
    return AnalysisResult(text, model_name, prediction, flag)
