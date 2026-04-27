from __future__ import annotations

from pathlib import Path

import torch

try:
    from .vit import (
        DEFAULT_IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        Preprocess,
        VisionInfer,
    )
except ImportError:
    from models.vit import (
        DEFAULT_IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        Preprocess,
        VisionInfer,
    )


class InternVisionModel(VisionInfer):
    def __init__(
        self,
        vit_engine_path: str | Path,
        stream: int,
        device: str | torch.device = "cuda",
    ):
        super().__init__(vit_engine_path, stream, str(device))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.infer(pixel_values)


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Preprocess",
    "VisionInfer",
    "InternVisionModel",
]
