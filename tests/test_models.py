"""
Unit tests for DeepLOB & Spatial-Temporal Transformer Models.
"""

import torch
import pytest

from src.models.deeplob import DeepLOB, SpatialTemporalTransformerLOB
from src.models.loss import FocalLoss, ClassWeightedCrossEntropy
from src.models.lightning_module import DeepLOBModule


def test_deeplob_forward_pass():
    """Verify DeepLOB output shape for standard (B, T, F) input."""
    batch_size = 4
    seq_len = 100
    num_features = 40
    num_classes = 3

    model = DeepLOB(in_features=num_features, seq_len=seq_len, num_classes=num_classes)
    x = torch.randn(batch_size, seq_len, num_features)
    
    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_transformer_forward_pass():
    """Verify SpatialTemporalTransformerLOB output shape."""
    batch_size = 4
    seq_len = 100
    num_features = 40
    num_classes = 3

    model = SpatialTemporalTransformerLOB(in_features=num_features, seq_len=seq_len, num_classes=num_classes)
    x = torch.randn(batch_size, seq_len, num_features)

    out = model(x)
    assert out.shape == (batch_size, num_classes)


def test_focal_loss():
    """Verify Focal Loss computation and gradient propagation."""
    batch_size = 8
    num_classes = 3
    
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 1, 2, 0], dtype=torch.long)

    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, targets)

    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_lightning_module_step():
    """Verify PyTorch Lightning Module training step."""
    batch_size = 4
    seq_len = 100
    num_features = 40
    
    module = DeepLOBModule(
        model_type="deeplob",
        in_features=num_features,
        seq_len=seq_len,
        loss_type="focal",
    )

    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    batch = (x, y)

    loss = module.training_step(batch, batch_idx=0)
    assert loss is not None
    assert loss.item() > 0
