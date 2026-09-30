"""BiLSTM baseline: architecture, per-epoch train/eval steps and checkpoint I/O.

Architecture::

    token ids -> Embedding -> BiLSTM (hidden=128, layers=2) -> Dropout(0.5) -> Linear -> logits
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from src import config
from src.data_utils import Vocabulary


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        num_classes: int = config.NUM_CLASSES,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        # Packing makes the LSTM skip padding, so the final hidden state is the
        # state at each sequence's real last token.
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # Last layer's forward (-2) and backward (-1) final states.
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(features))


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: BiLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = config.GRAD_CLIP,
) -> tuple[float, float]:
    """Run one training epoch; return ``(mean loss, accuracy)``."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: BiLSTM, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Evaluate without gradients; return ``(mean loss, accuracy)``."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for input_ids, lengths, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids, lengths)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def model_hparams(model: BiLSTM) -> dict:
    return {
        "vocab_size": model.embedding.num_embeddings,
        "pad_idx": model.embedding.padding_idx,
        "embedding_dim": model.embedding.embedding_dim,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "dropout": model.dropout.p,
        "num_classes": model.fc.out_features,
    }


def save_checkpoint(
    model: BiLSTM,
    vocab: Vocabulary,
    path: Path = config.BILSTM_CHECKPOINT_PATH,
    **extra,
) -> None:
    """Save weights together with vocab and hyperparameters (self-contained)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hparams": model_hparams(model),
            "vocab_itos": vocab.itos,
            "max_len": config.BILSTM_MAX_LEN,
            **extra,
        },
        path,
    )


def load_checkpoint(
    path: Path = config.BILSTM_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
) -> tuple[BiLSTM, Vocabulary, dict]:
    """Rebuild the model and vocabulary from a checkpoint (model in eval mode)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"BiLSTM checkpoint not found at {path}. "
            "Train it with notebooks/01_eda_and_baseline_bilstm.ipynb."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BiLSTM(**checkpoint["hparams"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, Vocabulary(checkpoint["vocab_itos"]), checkpoint
