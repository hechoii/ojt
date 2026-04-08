## Project Structure
ojt-w1/: Contains daily source code and learning materials (Day 1 to Day 5).

notebooks/: Jupyter notebooks documenting the experimentation and training process.

tests/: Unit tests to verify the mathematical correctness of the network layers.

src/train.py: The primary execution script.

## Setup
```bash
conda env create -f environment.yml
conda activate ojt-ai
```

## Run Training
```bash
python -m src.train
```

## Run Tests
```bash
pytest tests/ -v
# Expected: 4 passed
```

## Results
| Epoch | Train Loss | Train Acc |
|-------|-----------|-----------|
| 1     | 0.35      | 89.2%     |
| 5     | 0.08      | 97.1%     |