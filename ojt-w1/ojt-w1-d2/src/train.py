import torch

def preprocess_mnist(images: torch.Tensor) -> dict:
    """
    Args:
        images: (B, 1, 28, 28) uint8 [0,255]
    Returns:
        dict với keys: 'flat', 'mean', 'std'
    """
    B = images.shape[0]

    # TODO 1: convert sang float, normalize [0,1]
    x = images.float()/255.0

    # TODO 2: flatten → (B, 784)
    flat = x.reshape(B, -1)
    # x.revew(B,-1)

    # TODO 3: mean/std theo chiều batch (dim=0)
    mean = flat.mean(dim=0) # shape: (784,)
    std = flat.std(dim=0) # shape: (784,)

    return {"flat": flat, "mean": mean, "std": std}

# Test
dummy = torch.randint(0, 256, (64, 1, 28, 28), dtype=torch.uint8)
result = preprocess_mnist(dummy)
assert result["flat"].shape == (64, 784)
assert result["mean"].shape == (784,)
print("✅ Pass! flat:", result["flat"].shape)