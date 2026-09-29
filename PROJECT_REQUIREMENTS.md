# 🛒 E-commerce Review Sentiment & CS Flagging System

## 1. Tổng quan dự án (Project Overview)

Trong ngành thương mại điện tử, việc nắm bắt phản hồi của khách hàng là cực kỳ quan trọng. Dự án này xây dựng một hệ thống AI End-to-End giúp phân tích cảm xúc (Sentiment Analysis) từ các lượt đánh giá sản phẩm.

Hệ thống không chỉ phân loại **Tích cực / Tiêu cực** mà còn đóng vai trò như một bộ lọc thông minh:

> **Tự động nhận diện và gắn cờ (Flagging) các bình luận tiêu cực có tính chất khẩn cấp** như hàng giả, thái độ shipper tệ, sản phẩm hỏng hóc nặng, để chuyển ngay đến bộ phận Chăm sóc khách hàng (CSKH) xử lý.

---

## 2. Mục tiêu cốt lõi (Key Objectives)

1. **Model Benchmarking**  
   Xây dựng và so sánh hiệu năng giữa mô hình cơ sở tự code (**BiLSTM**) và mô hình học sâu hiện đại (**DistilBERT**).

2. **Business Logic (CS Flagging)**  
   Tích hợp logic xử lý nghiệp vụ để lọc các cảnh báo khẩn cấp dựa trên **từ khóa kết hợp với xác suất dự đoán của mô hình**.

3. **Interactive UI**  
   Đóng gói mô hình thành một Web App trực quan bằng **Gradio**.

4. **GitHub Portfolio Ready**  
   Tổ chức mã nguồn theo hướng **Software Engineering**, module hóa rõ ràng, dễ dàng clone, cài đặt và chạy thử.

---

## 3. Dataset & Tech Stack

### 3.1. Dataset

**Dataset đề xuất:**

- `amazon_polarity`
- Hoặc `amazon_reviews_multi`

Dataset được tải thông qua thư viện `datasets` của Hugging Face.

### 3.2. Quy mô dữ liệu

Để train nhanh và nghiệm thu:

- Tổng số sample: khoảng **20,000 dòng**
- Positive: **10,000**
- Negative: **10,000**
- Dataset phải được cân bằng giữa hai lớp.

### 3.3. Tech Stack

- **Python:** 3.10+
- **Deep Learning:** PyTorch
- **NLP / Transformers:** Hugging Face Transformers
- **Dataset:** Hugging Face Datasets
- **Data Processing:** Pandas
- **Machine Learning Utilities:** Scikit-learn
- **Visualization:** Matplotlib / Seaborn
- **Web UI:** Gradio

### 3.4. Môi trường

- **Kaggle Notebook:** dùng để R&D, train model và xuất file model (`.pt`, `.bin`, `.safetensors`)
- **VS Code / Cursor:** dùng để viết application code, tổ chức source code và hoàn thiện GitHub repository.

---

## 4. Cấu trúc thư mục yêu cầu (Project Structure)

Repository phải tuân thủ cấu trúc sau:

```text
ecommerce-sentiment-analyzer/
│
├── data/                   # (Đã ignore trong .gitignore)
│   └──                     # Dataset tải về / dữ liệu local
│
├── models/                 # File trọng số mô hình đã train
│   ├── bilstm_best.pt
│   └── ...                 # DistilBERT model / safetensors
│
├── notebooks/              # Jupyter Notebooks dùng cho R&D / Training / Evaluation
│   ├── 01_eda_and_baseline_bilstm.ipynb
│   └── 02_finetune_distilbert.ipynb
│
├── src/                    # Mã nguồn chính của ứng dụng
│   ├── __init__.py
│   ├── config.py           # Đường dẫn, hyperparameters, constants
│   ├── data_utils.py       # Làm sạch và tiền xử lý text
│   ├── flagging_logic.py   # Logic phát hiện keyword khẩn cấp
│   ├── model_bilstm.py     # Kiến trúc BiLSTM
│   └── inference.py        # Pipeline inference
│
├── app.py                  # File chính chạy giao diện Gradio
├── requirements.txt        # Danh sách dependencies
├── .gitignore
└── README.md               # Tài liệu mô tả, hướng dẫn và benchmark
```

---

## 5. Lộ trình thực hiện chi tiết (Phases of Execution)

# Phase 1: Data Preparation & EDA

### Mục tiêu

Chuẩn bị dataset sạch, cân bằng và có tập Train / Validation / Test độc lập để phục vụ toàn bộ quá trình huấn luyện và benchmark.

### Công việc

- Viết script / notebook tải dữ liệu từ Hugging Face.
- Phân tích dữ liệu:
  - Số lượng sample.
  - Phân bố nhãn.
  - Độ dài review.
  - Sample review tiêu biểu.
- Làm sạch dữ liệu:
  - Xóa HTML tags.
  - Chuẩn hóa chữ thường.
  - Xử lý khoảng trắng thừa.
  - Xử lý emoji theo chiến lược thống nhất.
- Cân bằng dữ liệu:
  - **10,000 Positive**
  - **10,000 Negative**
- Chia dữ liệu thành:
  - Train
  - Validation
  - Test
- Đảm bảo **Test Set độc lập** và được sử dụng chung cho cả BiLSTM và DistilBERT.

### Output

- Dataset đã được xử lý.
- EDA notebook hoàn chỉnh.
- Các biểu đồ thống kê cơ bản.
- Tập Train / Validation / Test có thể tái sử dụng.

---

# Phase 2: Baseline Model — BiLSTM

### Mục tiêu

Xây dựng một mô hình baseline bằng PyTorch để làm mốc benchmark cho DistilBERT.

### Kiến trúc yêu cầu

```text
Input Text
    ↓
Tokenization / Vocabulary
    ↓
Embedding
    ↓
BiLSTM
    ├── hidden_size = 128
    └── num_layers = 2
    ↓
Dropout
    └── p = 0.5
    ↓
Linear
    ↓
Binary Classification
```

### Công việc

- Tự xây dựng:
  - Vocabulary.
  - Token-to-index.
  - Index-to-token.
  - Unknown token.
  - Padding token.
- Tạo cơ chế padding cho batch.
- Xây dựng class `BiLSTM`.
- Viết training loop bằng PyTorch.
- Theo dõi:
  - Train Loss
  - Validation Loss
  - Train Accuracy
  - Validation Accuracy
- Lưu checkpoint có **Validation Loss thấp nhất**.

### Model output

```text
models/bilstm_best.pt
```

### Notebook

```text
notebooks/01_eda_and_baseline_bilstm.ipynb
```

Notebook phải chạy được từ trên xuống dưới mà không phát sinh runtime error.

---

# Phase 3: Production Model — DistilBERT

### Mục tiêu

Fine-tune mô hình Transformer hiện đại để sử dụng làm production model và benchmark với BiLSTM.

### Model

```text
distilbert-base-uncased
```

### Công việc

- Load tokenizer từ Hugging Face.
- Tokenize dataset.
- Fine-tune trên tập dữ liệu đã chuẩn bị.
- Có thể sử dụng:
  - Hugging Face `Trainer`
  - Hoặc custom PyTorch training loop.
- Theo dõi validation metrics.
- Save model + tokenizer vào thư mục `models/`.

### Output mong muốn

```text
models/
├── distilbert/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
```

### Notebook

```text
notebooks/02_finetune_distilbert.ipynb
```

### Acceptance target

> **DistilBERT Accuracy trên Test Set phải > 90%.**

Lưu ý: kết quả thực tế phụ thuộc vào random seed, sample size, preprocessing, training configuration và môi trường train.

---

# Phase 4: Model Evaluation & Benchmark

### Mục tiêu

Đánh giá hai mô hình trên **cùng một Test Set độc lập**.

### Mô hình cần đánh giá

- BiLSTM
- DistilBERT

### Metrics

Bắt buộc đo:

- Accuracy
- Precision
- Recall
- F1-Score

### Classification Report

Sử dụng:

```python
from sklearn.metrics import classification_report
```

In classification report cho cả hai mô hình.

### Confusion Matrix

Vẽ Confusion Matrix cho:

```text
BiLSTM
DistilBERT
```

Có thể sử dụng:

```python
matplotlib
seaborn
```

### Benchmark Table

README cần có bảng benchmark dạng tương tự:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| BiLSTM | TBD | TBD | TBD | TBD |
| DistilBERT | TBD | TBD | TBD | TBD |

> Không hard-code kết quả giả. Các chỉ số phải được lấy từ kết quả chạy thực tế trên Test Set.

---

# Phase 5: Business Logic — CS Flagging

## Mục tiêu

Không chỉ dự đoán sentiment, hệ thống còn phải xác định các review tiêu cực có tính chất **khẩn cấp** để chuyển cho CSKH.

## Điều kiện Flag

Review được gắn cờ khẩn cấp khi:

```text
Sentiment = Negative
        AND
Confidence > 80%
        AND
Text chứa ít nhất một Emergency Keyword
```

### Emergency Keywords đề xuất

```text
fake
scam
broken
refund
worst
terrible
dangerous
```

### Ví dụ logic

```python
if (
    sentiment == "Negative"
    and confidence > 0.80
    and contains_emergency_keyword(text)
):
    flag = "🔴 URGENT: TRANSFER TO CS"
else:
    flag = "✅ Normal"
```

### File phụ trách

```text
src/flagging_logic.py
```

### Yêu cầu thiết kế

Tách business logic khỏi model inference để:

- dễ test;
- dễ mở rộng keyword;
- dễ thay đổi threshold;
- không phụ thuộc trực tiếp vào UI.

---

# Phase 6: Gradio Interactive UI

### Mục tiêu

Đóng gói toàn bộ pipeline thành một Web App có thể chạy local.

### Input

Gradio UI phải có:

#### Review Text

```text
Textbox
```

Cho phép người dùng nhập nội dung review.

#### Model Selection

```text
Dropdown
```

Các lựa chọn:

```text
BiLSTM
DistilBERT
```

### Output

UI phải hiển thị:

#### 1. Sentiment

```text
Positive
Negative
```

#### 2. Confidence Score

Hiển thị dưới dạng ProgressBar hoặc component tương đương.

Ví dụ:

```text
Negative
Confidence: 92%
```

#### 3. CS Flag

Ví dụ:

```text
🔴 URGENT: TRANSFER TO CS
```

Hoặc:

```text
✅ Normal
```

### Inference Flow

```text
User Review
    ↓
Text Cleaning
    ↓
Selected Model
    ↓
Prediction
    ↓
Sentiment + Confidence
    ↓
CS Flagging Logic
    ↓
Gradio UI
```

### File phụ trách

```text
app.py
src/inference.py
src/data_utils.py
src/flagging_logic.py
```

### Acceptance Test

Input:

```text
The product arrived completely broken and smells dangerous, I want a refund now!
```

Expected:

```text
Sentiment: Negative
Confidence: > 80%   # phụ thuộc model thực tế
CS Flag: 🔴 URGENT: TRANSFER TO CS
```

---

# Phase 7: Hoàn thiện GitHub Repository

## README.md

README phải bao gồm các phần:

### 1. Project Title

Tên dự án:

```text
E-commerce Review Sentiment & CS Flagging System
```

### 2. Project Overview

Mô tả ngắn:

- bài toán;
- giải pháp;
- AI pipeline;
- CS flagging.

### 3. Features

Ví dụ:

- Sentiment classification.
- BiLSTM baseline.
- DistilBERT production model.
- Confidence score.
- Emergency keyword detection.
- Automatic CS flagging.
- Gradio Web UI.

### 4. Tech Stack

Liệt kê:

- Python
- PyTorch
- Transformers
- Hugging Face Datasets
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Gradio

### 5. Project Structure

Hiển thị tree của repository.

### 6. Installation

```bash
pip install -r requirements.txt
```

### 7. Run Application

```bash
python app.py
```

### 8. Benchmark

Thêm bảng kết quả thực tế:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BiLSTM | TBD | TBD | TBD | TBD |
| DistilBERT | TBD | TBD | TBD | TBD |

### 9. Demo

README cần chèn:

- Screenshot giao diện Gradio.
- Hoặc Demo GIF.

### 10. Example

Hiển thị input và expected behavior.

---

## 6. Tiêu chí nghiệm thu (Definition of Done)

### Code & Architecture

- [ ] Code không có runtime error.
- [ ] Source code được module hóa rõ ràng.
- [ ] Logic inference tách khỏi UI.
- [ ] Business logic tách khỏi model.
- [ ] Configuration tập trung trong `src/config.py`.

### Notebook

- [ ] `01_eda_and_baseline_bilstm.ipynb` chạy từ trên xuống dưới.
- [ ] `02_finetune_distilbert.ipynb` chạy từ trên xuống dưới.
- [ ] Có Markdown giải thích các bước chính.
- [ ] Có biểu đồ / metrics cần thiết.
- [ ] Không phụ thuộc vào biến được tạo thủ công ngoài notebook.

### Model

- [ ] BiLSTM train thành công.
- [ ] `models/bilstm_best.pt` được tạo.
- [ ] DistilBERT fine-tune thành công.
- [ ] Model và tokenizer được lưu vào `models/`.
- [ ] DistilBERT đạt Accuracy **> 90% trên Test Set**.

### Evaluation

- [ ] Hai model được đánh giá trên cùng Test Set.
- [ ] Có Accuracy.
- [ ] Có Precision.
- [ ] Có Recall.
- [ ] Có F1-Score.
- [ ] Có Classification Report.
- [ ] Có Confusion Matrix.
- [ ] Có benchmark table.

### CS Flagging

- [ ] Negative + confidence > 80% + emergency keyword → Flag urgent.
- [ ] Có message:

```text
🔴 URGENT: TRANSFER TO CS
```

- [ ] Logic được implement trong `src/flagging_logic.py`.

### Gradio UI

- [ ] Có textbox nhập review.
- [ ] Có dropdown chọn BiLSTM / DistilBERT.
- [ ] Có sentiment output.
- [ ] Có confidence output.
- [ ] Có CS flag output.
- [ ] `python app.py` chạy được.
- [ ] Web UI mở được tại localhost.

### GitHub

- [ ] Có `.gitignore`.
- [ ] Dataset local trong `data/` không commit lên Git.
- [ ] README đầy đủ.
- [ ] Có screenshot / GIF demo.
- [ ] Có benchmark thực tế.
- [ ] Repository có thể clone và setup theo README.

---

## 7. Suggested `requirements.txt`

Có thể bắt đầu với:

```txt
python>=3.10

torch
transformers
datasets
pandas
numpy
scikit-learn
matplotlib
seaborn
gradio
tqdm
```

> Phiên bản package cụ thể nên được pin sau khi hoàn tất môi trường train/inference để đảm bảo reproducibility, ví dụ `transformers==...`.

---

## 8. Reproducibility

Để kết quả benchmark ổn định và dễ tái tạo:

- Cố định random seed.
- Ghi lại hyperparameters.
- Ghi rõ số lượng dataset sử dụng.
- Ghi rõ train/validation/test split.
- Ghi lại version Python và các thư viện chính.
- Lưu tokenizer cùng model.
- Không sử dụng Test Set trong quá trình tuning.

Ví dụ:

```python
SEED = 42
```

---

## 9. Suggested Configuration

Các hyperparameters có thể tập trung trong:

```text
src/config.py
```

Ví dụ:

```python
SEED = 42

NUM_CLASSES = 2

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5

FLAG_CONFIDENCE_THRESHOLD = 0.80
```

DistilBERT có thể có configuration riêng, ví dụ:

```python
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
```

---

## 10. Recommended Testing Cases

Ngoài acceptance test chính, nên kiểm tra thêm các trường hợp sau.

### Case 1 — Positive Review

```text
The product is excellent. Great quality and fast delivery!
```

Expected:

```text
Sentiment: Positive
CS Flag: ✅ Normal
```

### Case 2 — Negative nhưng không khẩn cấp

```text
The product quality is disappointing. I don't really like it.
```

Expected:

```text
Sentiment: Negative
CS Flag: ✅ Normal
```

### Case 3 — Negative + Emergency Keyword

```text
This item looks fake and I want a refund immediately.
```

Expected:

```text
Sentiment: Negative
CS Flag: 🔴 URGENT: TRANSFER TO CS
```

### Case 4 — Emergency Keyword nhưng sentiment không Negative

```text
The refund process was simple and the support team was great.
```

Expected:

```text
Sentiment: Positive
CS Flag: ✅ Normal
```

### Case 5 — Original Acceptance Test

```text
The product arrived completely broken and smells dangerous, I want a refund now!
```

Expected:

```text
Sentiment: Negative
CS Flag: 🔴 URGENT: TRANSFER TO CS
```

---

## 11. End-to-End Architecture

```text
                  ┌─────────────────────┐
                  │   Amazon Reviews    │
                  │   Hugging Face      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Preparation    │
                  │ Cleaning / Balance  │
                  └──────────┬──────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
          ┌────────────────┐   ┌──────────────────┐
          │    BiLSTM      │   │    DistilBERT    │
          │    Baseline    │   │   Production     │
          └───────┬────────┘   └────────┬─────────┘
                  │                     │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   Model Inference   │
                  │ Sentiment +         │
                  │ Confidence          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   CS Flagging Logic │
                  │ Negative + >80% +   │
                  │ Emergency Keyword   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Gradio UI       │
                  │ Review / Model /    │
                  │ Sentiment / Flag    │
                  └─────────────────────┘
```

---

## 12. Final Deliverables

Sau khi hoàn thành, repository cần có tối thiểu:

```text
ecommerce-sentiment-analyzer/
├── data/
├── models/
│   ├── bilstm_best.pt
│   └── distilbert/
├── notebooks/
│   ├── 01_eda_and_baseline_bilstm.ipynb
│   └── 02_finetune_distilbert.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── flagging_logic.py
│   ├── model_bilstm.py
│   └── inference.py
├── app.py
├── requirements.txt
├── .gitignore
├── PROJECT_REQUIREMENTS.md
└── README.md
```

### Final checklist

- [ ] Dataset prepared & balanced.
- [ ] EDA completed.
- [ ] BiLSTM implemented and trained.
- [ ] DistilBERT fine-tuned.
- [ ] DistilBERT Test Accuracy > 90%.
- [ ] Both models benchmarked on the same Test Set.
- [ ] Confusion Matrices generated.
- [ ] Classification Reports generated.
- [ ] CS flagging logic implemented.
- [ ] Gradio UI completed.
- [ ] Model dropdown works.
- [ ] Confidence displayed.
- [ ] Urgent review is correctly flagged.
- [ ] README completed.
- [ ] Screenshot / GIF added.
- [ ] `python app.py` works locally.
- [ ] Repository is ready to publish on GitHub.
