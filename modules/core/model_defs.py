"""
modules/core/model_defs.py
==========================
Network architectures as defined in Thesis3.ipynb.

The ANN is the three-layer MLP from the notebook's final configuration:
128 -> 64 -> 1 neurons, ReLU activations, dropout 0.6/0.4, sigmoid output.

PyTorch is imported lazily (PEP 562 ``__getattr__``) so that importing this
module — and therefore the whole app — works before ``torch`` is installed
or the model is trained. ``CognitiveImpairmentANN`` materialises the first
time it is actually used (only inside the prediction/training paths).
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily define CognitiveImpairmentANN on first use."""
    if name != "CognitiveImpairmentANN":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to build the ANN. Install it with "
            "`pip install torch`, then train the model with "
            "`python scripts/train_models.py`."
        ) from exc

    class CognitiveImpairmentANN(nn.Module):
        """Three-layer MLP used as the deep-learning arm of the hybrid ensemble."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    globals()["CognitiveImpairmentANN"] = CognitiveImpairmentANN
    return CognitiveImpairmentANN
