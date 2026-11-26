"""PyTorch Dataset reading Cassava images from disk."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..config import IMAGE_DIR


class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, image_dir: Path = IMAGE_DIR):
        self.ids = df.image_id.tolist()
        self.labels = df.label.to_numpy()
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with Image.open(self.image_dir / self.ids[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(self.labels[i])
