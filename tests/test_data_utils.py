import pandas as pd
import torch

from src import config
from src.data_utils import (
    PadCollator,
    Vocabulary,
    balance_dataset,
    clean_text,
    preprocess_dataframe,
    split_dataset,
    tokenize,
)


def test_clean_text_html_case_whitespace_urls():
    raw = "  Great <br/>PRODUCT!&amp; see http://x.com   now "
    assert clean_text(raw) == "great product!& see now"


def test_clean_text_emoji_strategy():
    assert clean_text("Loved it 😍👍") == "loved it love good"
    assert clean_text("Broken 😡🚀") == "broken angry"  # unmapped emoji removed


def test_clean_text_is_idempotent_and_handles_non_str():
    text = clean_text("Hello <b>World</b> 😀")
    assert clean_text(text) == text
    assert clean_text(None) == ""


def test_tokenize_keeps_negations_and_marks():
    assert tokenize("i don't like it!") == ["i", "don't", "like", "it", "!"]


def _toy_df(n_per_class=20):
    return pd.DataFrame(
        {
            "text": [f"good review {i}" for i in range(n_per_class)]
            + [f"bad review {i}" for i in range(n_per_class + 5)],
            "label": [1] * n_per_class + [0] * (n_per_class + 5),
        }
    )


def test_preprocess_drops_empty_and_duplicates():
    df = pd.DataFrame({"text": ["A", "a", "<br>", "b"], "label": [1, 1, 0, 0]})
    assert preprocess_dataframe(df)["text"].tolist() == ["a", "b"]


def test_balance_and_split_are_stratified_and_disjoint():
    df = balance_dataset(_toy_df(), samples_per_class=20, seed=0)
    assert df["label"].value_counts().to_dict() == {0: 20, 1: 20}

    splits = split_dataset(df, val_size=0.1, test_size=0.1, seed=0)
    assert [len(splits[s]) for s in ("train", "val", "test")] == [32, 4, 4]
    for split in splits.values():
        assert split["label"].value_counts().nunique() == 1  # balanced
    texts = [set(splits[s]["text"]) for s in ("train", "val", "test")]
    assert not (texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2])


def test_vocabulary_special_tokens_and_unknowns():
    vocab = Vocabulary.build(["good good bad", "good bad ok"], max_size=10, min_freq=2)
    assert vocab.itos[:2] == [config.PAD_TOKEN, config.UNK_TOKEN]
    assert "ok" not in vocab.stoi  # below min_freq
    ids = vocab.encode("good unseen")
    assert ids == [vocab.stoi["good"], vocab.unk_idx]
    assert vocab.encode("") == [vocab.unk_idx]
    assert vocab.decode(ids) == ["good", config.UNK_TOKEN]


def test_collator_pads_batch_and_returns_lengths():
    batch = [(torch.tensor([2, 3, 4]), 1), (torch.tensor([5]), 0)]
    input_ids, lengths, labels = PadCollator(pad_idx=0)(batch)
    assert input_ids.tolist() == [[2, 3, 4], [5, 0, 0]]
    assert lengths.tolist() == [3, 1]
    assert labels.tolist() == [1, 0]
