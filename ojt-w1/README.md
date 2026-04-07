# OJT W1 — FC Network on MNIST

Simple fully-connected network trained on MNIST, built from scratch as part of
the ABIM OJT AI Engineer program (Week 1).

## Setup
```bash
conda env create -f environment.yml
conda activate ojt-ai
```

## Run Training
``` in week 1, from day 1 to day 5, Training model in notebooks\.ipynb_checkpoints\*.ipynb
Every code how to train and result can be seen in that file
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