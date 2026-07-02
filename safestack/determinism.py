"""Global seed handling for reproducible values (ADR-0004).

Safe to call on the torch-free mock path: numpy/torch/transformers are seeded only when
importable. `use_deterministic_algorithms` is best-effort because some MPS/CPU ops lack
deterministic implementations and raise.
"""

from __future__ import annotations

import random


def set_seeds(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except ImportError:
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass
