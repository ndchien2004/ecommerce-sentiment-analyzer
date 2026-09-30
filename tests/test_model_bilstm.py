import torch

from src import config
from src.data_utils import PadCollator, Vocabulary
from src.model_bilstm import BiLSTM, load_checkpoint, save_checkpoint


def _batch():
    batch = [(torch.tensor([2, 3, 4]), 1), (torch.tensor([5]), 0)]
    return PadCollator(pad_idx=0)(batch)


def test_forward_returns_logits_per_class():
    input_ids, lengths, _ = _batch()
    logits = BiLSTM(vocab_size=10, pad_idx=0)(input_ids, lengths)
    assert logits.shape == (2, config.NUM_CLASSES)


def test_padding_does_not_change_output():
    model = BiLSTM(vocab_size=10, pad_idx=0).eval()
    short = model(torch.tensor([[5]]), torch.tensor([1]))
    padded = model(torch.tensor([[5, 0, 0]]), torch.tensor([1]))
    assert torch.allclose(short, padded, atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    vocab = Vocabulary.build(["good good bad bad"], min_freq=1)
    model = BiLSTM(vocab_size=len(vocab), pad_idx=vocab.pad_idx).eval()
    path = tmp_path / "bilstm.pt"
    save_checkpoint(model, vocab, path, epoch=3, val_loss=0.25)

    loaded, loaded_vocab, checkpoint = load_checkpoint(path)
    assert loaded_vocab.itos == vocab.itos
    assert checkpoint["epoch"] == 3
    input_ids, lengths, _ = _batch()
    input_ids = input_ids.clamp(max=len(vocab) - 1)
    assert torch.allclose(model(input_ids, lengths), loaded(input_ids, lengths))
