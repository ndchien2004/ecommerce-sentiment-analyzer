"""Text cleaning, dataset preparation and BiLSTM tokenization utilities.

The same ``clean_text`` function is used for training (both models), evaluation
and the Gradio app, so the model always sees text preprocessed the same way.
"""

from __future__ import annotations

import html
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"  # flags
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f900-\U0001faff"  # supplemental symbols
    "☀-➿"  # misc symbols & dingbats
    "‍️"  # zero-width joiner, variation selector
    "]+"
)

# Emoji strategy: sentiment-bearing emojis are translated into words so both
# models can use the signal; every other emoji is removed.
EMOJI_TO_WORD = {
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "😊": "happy",
    "🙂": "happy", "😍": "love", "🥰": "love", "❤": "love", "💯": "perfect",
    "👍": "good", "👌": "good", "🎉": "great", "⭐": "star",
    "🙁": "sad", "☹": "sad", "😞": "sad", "😢": "sad", "😭": "sad",
    "😠": "angry", "😡": "angry", "🤬": "angry", "👎": "bad", "🤮": "disgusting",
    "💩": "bad", "😤": "angry",
}  # fmt: skip


def replace_emojis(text: str) -> str:
    for emoji, word in EMOJI_TO_WORD.items():
        text = text.replace(emoji, f" {word} ")
    return EMOJI_RE.sub(" ", text)


def clean_text(text: str) -> str:
    """Normalise a raw review: HTML, URLs, emojis, case and whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = replace_emojis(text)
    text = text.lower()
    return WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Dataset preparation (Phase 1)
# ---------------------------------------------------------------------------

SPLIT_NAMES = ("train", "val", "test")


def load_raw_dataset() -> pd.DataFrame:
    """Download one shard of amazon_polarity and return ``text``/``label`` columns.

    Title and body are joined because the title often carries the strongest
    sentiment cue ("Total waste of money").
    """
    from datasets import load_dataset

    # Split-size verification compares against the full dataset's metadata, which
    # a single shard never matches, so it is disabled.
    ds = load_dataset(
        config.DATASET_NAME,
        data_files=config.DATASET_DATA_FILES,
        split="train",
        verification_mode="no_checks",
    )
    df = ds.to_pandas()
    df["text"] = df["title"].fillna("").str.strip() + ". " + df["content"].fillna("")
    return df[["text", "label"]]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, then drop empty and duplicated reviews."""
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def balance_dataset(
    df: pd.DataFrame,
    samples_per_class: int = config.SAMPLES_PER_CLASS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Sample exactly ``samples_per_class`` rows from every label, then shuffle."""
    parts = [
        group.sample(n=samples_per_class, random_state=seed)
        for _, group in df.groupby("label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split."""
    from sklearn.model_selection import train_test_split

    # Absolute sizes avoid float rounding (e.g. 0.1 / 0.9 * 18000 -> 2001 rows).
    n_test = round(len(df) * test_size)
    n_val = round(len(df) * val_size)
    train_val, test = train_test_split(
        df, test_size=n_test, stratify=df["label"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=n_val, stratify=train_val["label"], random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _split_path(name: str):
    return config.PROCESSED_DATA_DIR / f"{name}.csv"


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        splits[name].to_csv(_split_path(name), index=False)


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(_split_path(name), keep_default_na=False) for name in SPLIT_NAMES}


def prepare_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    """Return the train/val/test splits, building and caching them if needed.

    Splits are cached as CSV in ``data/processed`` so both notebooks, the
    evaluation script and any re-run use exactly the same independent test set.
    Building is deterministic (fixed seed), so a fresh machine rebuilds the
    identical split.
    """
    if not force and all(_split_path(name).exists() for name in SPLIT_NAMES):
        return load_splits()

    splits = split_dataset(balance_dataset(preprocess_dataframe(load_raw_dataset())))
    save_splits(splits)
    return splits


# ---------------------------------------------------------------------------
# BiLSTM tokenization, vocabulary and batching (Phase 2)
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def tokenize(text: str) -> list[str]:
    """Word-level tokenizer for already-cleaned text (keeps negations like don't)."""
    return TOKEN_RE.findall(text)


class Vocabulary:
    """Token <-> index mapping with ``<pad>`` (index 0) and ``<unk>`` (index 1)."""

    def __init__(self, itos: Sequence[str]):
        self.itos: list[str] = list(itos)
        self.stoi: dict[str, int] = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_idx = self.stoi[config.PAD_TOKEN]
        self.unk_idx = self.stoi[config.UNK_TOKEN]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_size: int = config.MAX_VOCAB_SIZE,
        min_freq: int = config.MIN_FREQ,
    ) -> Vocabulary:
        counter = Counter(token for text in texts for token in tokenize(text))
        specials = [config.PAD_TOKEN, config.UNK_TOKEN]
        words = [
            word
            for word, freq in counter.most_common(max_size - len(specials))
            if freq >= min_freq
        ]
        return cls(specials + words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = config.BILSTM_MAX_LEN) -> list[int]:
        ids = [self.stoi.get(token, self.unk_idx) for token in tokenize(text)[:max_len]]
        # pack_padded_sequence rejects zero-length sequences.
        return ids or [self.unk_idx]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids if i != self.pad_idx]


class ReviewDataset(Dataset):
    """Encodes reviews on the fly into index tensors for the BiLSTM."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocabulary,
        max_len: int = config.BILSTM_MAX_LEN,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), int(self.labels[idx])


def pad_batch(
    sequences: Sequence[torch.Tensor], pad_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length sequences; return ``(padded, lengths)``."""
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = torch.full((len(sequences), int(lengths.max())), pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded, lengths


class PadCollator:
    """``collate_fn`` for DataLoader (a class so it pickles for worker processes)."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: Sequence[tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        padded, lengths = pad_batch(sequences, self.pad_idx)
        return padded, lengths, torch.tensor(labels, dtype=torch.long)
