"""Central configuration: paths, dataset settings, hyperparameters and business rules.

Every other module reads its constants from here so experiments stay reproducible
and a single edit changes behaviour everywhere (training, evaluation, app).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
BILSTM_CHECKPOINT_PATH = MODELS_DIR / "bilstm_best.pt"
DISTILBERT_DIR = MODELS_DIR / "distilbert"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "fancyzhx/amazon_polarity"
# One parquet shard (~900k reviews) is plenty to draw a balanced 20k sample
# and avoids downloading the full 3.6M-row training set.
DATASET_DATA_FILES = {"train": "amazon_polarity/train-00000-of-00004.parquet"}

SAMPLES_PER_CLASS = 10_000
VAL_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 2
ID2LABEL = {0: "Negative", 1: "Positive"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

# ---------------------------------------------------------------------------
# BiLSTM baseline
# ---------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_VOCAB_SIZE = 20_000
MIN_FREQ = 2
BILSTM_MAX_LEN = 256

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
GRAD_CLIP = 1.0

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5

# ---------------------------------------------------------------------------
# DistilBERT production model
# ---------------------------------------------------------------------------
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
DISTILBERT_BATCH_SIZE = 16
DISTILBERT_LEARNING_RATE = 2e-5
DISTILBERT_NUM_EPOCHS = 2
DISTILBERT_WEIGHT_DECAY = 0.01
DISTILBERT_WARMUP_RATIO = 0.1

# ---------------------------------------------------------------------------
# CS flagging (business rules)
# ---------------------------------------------------------------------------
FLAG_CONFIDENCE_THRESHOLD = 0.80
EMERGENCY_KEYWORDS = (
    "fake",
    "scam",
    "broken",
    "refund",
    "worst",
    "terrible",
    "dangerous",
)
URGENT_FLAG = "🔴 URGENT: TRANSFER TO CS"
NORMAL_FLAG = "✅ Normal"
