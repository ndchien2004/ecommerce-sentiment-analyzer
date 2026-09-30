"""Mini-Training Simulator: follow 2 reviews through ONE training step (1 epoch, 1 batch).

Run from a terminal:
    python scripts/training_simulator.py                    # BiLSTM, tiny sizes so tensors are readable
    python scripts/training_simulator.py --model distilbert # real distilbert-base-uncased (CPU is fine)

Standalone: needs only torch + rich (and transformers for --model distilbert).
"""

import argparse
import math
import os
import re

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

console = Console()
torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=110)

REVIEWS = [
    ("Great quality, I love it!", 1),
    ("The item arrived broken, I want a refund.", 0),
]
LABEL_NAMES = {0: "Negative", 1: "Positive"}

# Tiny BiLSTM sizes so every tensor fits on screen (the real project uses 128 / 128).
EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS = 8, 4, 2


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def step(number: int, title: str, analogy: str) -> None:
    console.print()
    console.rule(f"[bold cyan]BƯỚC {number} - {title}")
    console.print(f"[italic]💡 {analogy}\n")


def show(name: str, tensor: torch.Tensor, note: str = "", values: bool = True) -> None:
    console.print(f"[bold yellow]{name}[/]  shape = [bold green]{tuple(tensor.shape)}[/]  [dim]{note}")
    if values:
        console.print(tensor.detach(), highlight=False)


# ---------------------------------------------------------------------------
# BiLSTM path (same architecture as src/model_bilstm.py, just smaller)
# ---------------------------------------------------------------------------
class TinyBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=pad_idx)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, 2)

    def forward(self, input_ids, lengths, verbose: bool = False):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        features = torch.cat((h_n[-2], h_n[-1]), dim=1)  # last layer: forward + backward
        logits = self.fc(self.dropout(features))
        if verbose:
            outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
            show("3.1 Embedding", embedded, "(batch, seq_len, embedding_dim): mỗi token -> 1 vector 8 số",
                 values=False)
            console.print(f"    vector của token đầu tiên câu 1: {embedded[0, 0].detach()}\n", highlight=False)
            show("3.2 LSTM outputs (mọi bước thời gian)", outputs,
                 f"(batch, seq_len, 2*hidden={2 * HIDDEN_SIZE}): chiều xuôi + chiều ngược", values=False)
            console.print("    câu 1 (ngắn hơn): các hàng cuối = 0 vì LSTM KHÔNG đọc phần <pad>", highlight=False)
            console.print(outputs[0].detach(), highlight=False)
            console.print()
            show("3.3 h_n (trạng thái ẩn cuối)", h_n,
                 f"(num_layers*2 hướng={NUM_LAYERS * 2}, batch, hidden): 'bản tóm tắt' của từng tầng/hướng",
                 values=False)
            show("3.4 features = concat(h_n[-2], h_n[-1])", features,
                 "(batch, 2*hidden): 1 vector tóm tắt cho CẢ câu")
            show("3.5 Linear -> logits", logits, "(batch, num_classes): điểm thô cho [Negative, Positive]")
        return logits


def tokenize_bilstm(texts):
    tokens = [re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[!?]", t.lower()) for t in texts]
    itos = ["<pad>", "<unk>"] + sorted({tok for sent in tokens for tok in sent})
    stoi = {tok: i for i, tok in enumerate(itos)}
    ids = [[stoi[tok] for tok in sent] for sent in tokens]
    lengths = torch.tensor([len(x) for x in ids])
    input_ids = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)  # 0 = <pad>
    for i, seq in enumerate(ids):
        input_ids[i, : len(seq)] = torch.tensor(seq)

    for i, sent in enumerate(tokens):
        console.print(f"[bold]Câu {i + 1} tokens ({len(sent)}):[/] {sent}")
    console.print(f"\n[bold]Vocabulary[/] ({len(itos)} từ): {dict(list(stoi.items())[:8])} ...\n")
    show("input_ids", input_ids, "(batch, seq_len): câu ngắn hơn được đệm số 0 (<pad>) ở cuối")
    show("attention_mask (= input_ids != pad)", (input_ids != 0).long(),
         "1 = token thật, 0 = đệm. BiLSTM dùng 'lengths' thay cho mask")
    show("lengths", lengths, "độ dài thật -> pack_padded_sequence bỏ qua phần đệm")
    return TinyBiLSTM(len(itos), pad_idx=0), (input_ids, lengths)


# ---------------------------------------------------------------------------
# DistilBERT path (real pretrained weights, new 2-class head)
# ---------------------------------------------------------------------------
def tokenize_distilbert(texts):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    enc = tokenizer(texts, padding=True, return_tensors="pt")

    for i in range(len(texts)):
        console.print(f"[bold]Câu {i + 1} WordPiece:[/] {tokenizer.convert_ids_to_tokens(enc['input_ids'][i])}")
    console.print()
    show("input_ids", enc["input_ids"], "(batch, seq_len): [CLS]=101, [SEP]=102, [PAD]=0")
    show("attention_mask", enc["attention_mask"], "1 = token thật, 0 = [PAD] -> self-attention bỏ qua")
    return model, (enc["input_ids"], enc["attention_mask"])


def forward_distilbert(model, input_ids, attention_mask, verbose: bool = False):
    out = model.distilbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    cls = out.last_hidden_state[:, 0]  # vector of the [CLS] token summarises the sentence
    hidden = model.dropout(nn.functional.relu(model.pre_classifier(cls)))
    logits = model.classifier(hidden)
    if verbose:
        show("3.1 Embedding (token + position)", out.hidden_states[0], "(batch, seq_len, 768)", values=False)
        console.print(f"    Qua {len(out.hidden_states) - 1} tầng Transformer, mỗi tầng giữ nguyên shape:")
        for i, h in enumerate(out.hidden_states[1:], 1):
            console.print(f"      tầng {i}: {tuple(h.shape)}", highlight=False)
        show("3.2 Vector [CLS]", cls, "(batch, 768): đại diện cho cả câu", values=False)
        show("3.3 pre_classifier + ReLU + Dropout", hidden, "(batch, 768)", values=False)
        show("3.4 classifier -> logits", logits, "(batch, num_classes)")
    return logits


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def main(model_type: str) -> None:
    texts = [t for t, _ in REVIEWS]
    labels = torch.tensor([y for _, y in REVIEWS])

    step(1, "DỮ LIỆU THÔ", "Giống xấp phiếu góp ý viết tay: máy chưa hiểu chữ, chỉ thấy chuỗi ký tự.")
    table = Table("#", "Review", "Label")
    for i, (text, y) in enumerate(REVIEWS, 1):
        table.add_row(str(i), text, f"{y} ({LABEL_NAMES[y]})")
    console.print(table)
    show("labels", labels, "(batch,): đáp án đúng")

    step(2, "TOKENIZER + BATCHING",
         "Giống cắt câu thành mảnh ghép rồi tra từ điển lấy số thứ tự; câu ngắn được kê thêm 'giấy lót' "
         "cho bằng câu dài để xếp chung một khay (batch).")
    if model_type == "bilstm":
        model, inputs = tokenize_bilstm(texts)
        forward = lambda verbose=False: model(*inputs, verbose=verbose)  # noqa: E731
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        watched_name = "fc.weight"
    else:
        model, inputs = tokenize_distilbert(texts)
        forward = lambda verbose=False: forward_distilbert(model, *inputs, verbose=verbose)  # noqa: E731
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        watched_name = "classifier.weight"
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[bold]Model:[/] {model_type} với [bold]{n_params:,}[/] tham số (trọng số)")

    with torch.no_grad():
        model.eval()
        loss_before = nn.functional.cross_entropy(forward(), labels).item()

    step(3, "FORWARD PASS",
         "Giống dây chuyền nhà máy: mỗi trạm biến đổi 'bán thành phẩm' (tensor) và đổi hình dạng của nó.")
    model.train()  # dropout active, exactly like real training
    logits = forward(verbose=True)

    step(4, "TÍNH LOSS",
         "Giống giáo viên chấm bài: so đáp án của học sinh với đáp án đúng, càng tự tin mà sai càng bị trừ nặng.")
    probs = torch.softmax(logits, dim=1)
    show("softmax(logits) -> xác suất", probs, "mỗi hàng cộng lại = 1")
    table = Table("#", "Logits [Neg, Pos]", "P(Neg)", "P(Pos)", "Dự đoán", "Nhãn thật", "loss_i = -ln P(nhãn thật)")
    for i in range(len(texts)):
        p_true = probs[i, labels[i]].item()
        pred = LABEL_NAMES[int(probs[i].argmax())]
        table.add_row(str(i + 1), str([round(v, 3) for v in logits[i].tolist()]),
                      f"{probs[i, 0]:.3f}", f"{probs[i, 1]:.3f}", pred, LABEL_NAMES[int(labels[i])],
                      f"-ln({p_true:.3f}) = {-math.log(p_true):.4f}")
    console.print(table)
    loss = nn.functional.cross_entropy(logits, labels)
    console.print(f"[bold]CrossEntropy loss (trung bình batch) = {loss.item():.4f}[/]  "
                  f"[dim](đoán mò 50/50 cho loss = ln 2 ≈ 0.6931)")

    step(5, "BACKWARD PASS + OPTIMIZER",
         "Giống sau khi chấm bài, truy ngược xem 'lỗi' do trạm nào gây ra (gradient) rồi vặn mỗi núm "
         "một chút theo hướng giảm lỗi (optimizer.step).")
    optimizer.zero_grad()
    loss.backward()

    grads = Table("Tham số", "Shape", "Số lượng", "‖gradient‖")
    for name, p in model.named_parameters():
        if model_type == "bilstm" or not name.startswith("distilbert.transformer.layer."):
            grads.add_row(name, str(tuple(p.shape)), f"{p.numel():,}", f"{p.grad.norm():.5f}")
    if model_type == "distilbert":
        grads.add_row("distilbert.transformer.layer.0-5.*", "...", "...", "(ẩn bớt cho gọn)")
    console.print(grads)

    # Watch the entry with the largest gradient (dropout may have zeroed others).
    weight = dict(model.named_parameters())[watched_name]
    row, col = divmod(int(weight.grad.abs().argmax()), weight.shape[1])
    before, grad, lr = weight[row, col].item(), weight.grad[row, col].item(), optimizer.param_groups[0]["lr"]
    optimizer.step()
    after = weight[row, col].item()
    console.print(f"\nTheo dõi 1 trọng số [bold]{watched_name}[{row},{col}][/] (gradient lớn nhất):")
    console.print(f"  w_mới = w_cũ - lr × grad = {before:.5f} - {lr} × ({grad:.5f}) = "
                  f"{before - lr * grad:.5f}   | thực tế sau optimizer.step(): [bold green]{after:.5f}[/]")

    with torch.no_grad():
        model.eval()
        loss_after = nn.functional.cross_entropy(forward(), labels).item()
    console.print(f"\n[bold]Loss (eval, không dropout) trước bước học: {loss_before:.4f} -> "
                  f"sau 1 bước: [green]{loss_after:.4f}[/][/]")
    console.rule("[bold cyan]HẾT 1 EPOCH (dataset 2 câu, batch_size = 2 -> 1 epoch = 1 batch = 1 bước cập nhật)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["bilstm", "distilbert"], default="bilstm")
    main(parser.parse_args().model)
