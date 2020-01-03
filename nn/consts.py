from pathlib import Path


IM_HEIGHT = 200
IM_WIDTH = 300

IMG_ROOT = Path("data/raw").absolute()
LABELS_FILE = IMG_ROOT / "labels.json"

CLASSES = ("relax", "tense", "unfocused")
MODEL_PATH = Path("data/models/test.hdf5")
