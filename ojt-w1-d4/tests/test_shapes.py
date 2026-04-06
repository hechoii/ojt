import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('./src/models')
from src.models.fc_net import FCNet

@pytest.fixture
def mnist_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=32)

@pytest.fixture
def model():
    return FCNet()

def test_mnist_batch_shape(mnist_loader):
    imgs, labels = next(iter(mnist_loader))
    assert imgs.shape == (32, 1, 28, 28), f"Expected (32,1,28,28), got {imgs.shape}"
    assert labels.shape == (32,), f"Expected (32,), got {labels.shape}"
    
def test_model_output_shape(model, mnist_loader):
    model.eval()
    imgs, _ = next(iter(mnist_loader))
    with torch.no_grad():
        outputs = model(imgs)
    assert outputs.shape == (32, 10), f"Expected (32,10), got {outputs.shape}"
    
def test_model_output_has_no_nan(model, mnist_loader):
    model.eval()
    imgs, _ = next(iter(mnist_loader))
    with torch.no_grad():
        outputs = model(imgs)
    assert not torch.isnan(outputs).any(), f"Model output contains NaN values"
    
def test_model_output_finite(model, mnist_loader):
    model.eval()
    imgs, _ = next(iter(mnist_loader))
    with torch.no_grad():
        out = model(imgs)
    assert torch.isfinite(out).all(), "Model output contains Inf values"
    
