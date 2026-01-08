import torch

class LinearNormalizer:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor to [-1, 1] range."""
        return 2.0 * (x - self.min_val) / (self.max_val - self.min_val) - 1.0
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor from [-1, 1] range back to original range."""
        return 0.5 * (x + 1.0) * (self.max_val - self.min_val) + self.min_val