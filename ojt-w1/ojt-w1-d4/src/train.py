from src.data.dataset import CSVDataset, ImageFolderDataset

# Tạo dummy CSV để test
import pandas as pd
from pathlib import Path
import numpy as np

print("\nTesting ImageFolderDataset...")

# Create dummy ImageFolder to test
data = np.random.rand(100, 3)  # 100 samples, 3 features each
labels = np.random.randint(0, 2, size=100)  # Binary labels for each sample
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
df["labels"] = labels 
print(df.head())
print(f"Dataset: {len(df)} samples")

print("\nTesting CSVDataset...")

# Create dummy CSV to test
data = np.random.randn(200, 4)
labels = np.random.randint(0, 3, 200)
df = pd.DataFrame(data, columns=["f1","f2","f3","f4"])
df["label"] = labels

output_path = Path("tests/tmp/dummy.csv")
output_path.parent.mkdir(parents=True, exist_ok=True) # create folder if not exist
df.to_csv(output_path, index=False)

ds = CSVDataset("./tests/tmp/dummy.csv", ["f1","f2","f3","f4"], "label")
print(f"✅ Dataset: {len(ds)} samples")
x, y = ds[0]
print(f"   x: {x.shape}, y: {y}")
