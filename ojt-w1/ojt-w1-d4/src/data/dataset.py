import os
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,) -> None:
        self.root = Path(root)
        self.transform = transform
        
        # get all image file paths
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # gather all image paths and their labels
        self.samples: list[Tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.classes_to_idx[class_name]))
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, self.classes_to_idx[class_name]))
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
class CSVDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols: list, label_col: str) -> None:
        df = pd.read_csv(csv_path)
        assert df[feature_cols + [label_col]].isna().sum().sum() == 0, "Dataset contains NaN values!"
        
        self.x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]