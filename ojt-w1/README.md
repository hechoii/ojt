# OJT W1 — FC Network on MNIST

Simple fully-connected network trained on MNIST, built from scratch as part of
the ABIM OJT AI Engineer program (Week 1).

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
# Expected: Epoch 5 | Loss: ~0.08 | Acc: ~97%
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