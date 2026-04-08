from src.models.fc_net import FCNet
import torch

model = FCNet()
dummy = torch.randn(32,1,28,28)
output = model(dummy)
assert output.shape == (32,10)
print("✅ Forward pass OK, shape:", output.shape)