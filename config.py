from pathlib import Path

# プロジェクト全体で共有する設定
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
TEST_DIR = ROOT / "test"
MODEL_PATH = ROOT / "dog_cat_model.keras"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
REQUIREMENTS = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
EPOCHS = 5
