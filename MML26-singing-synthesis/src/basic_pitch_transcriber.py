"""Wrapper around BasicPitchTorch for the OAF training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .basic_pitch_model import BasicPitchTorch


class BasicPitchTranscriber(nn.Module):
    """Basic Pitch transcriber returning onset/frame/contour logits (127 pitch bins)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = BasicPitchTorch()

    def forward(
        self, audio: torch.Tensor, return_contour: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio: (batch, time) or (batch, 1, time) at SAMPLE_RATE (16 kHz).
            return_contour: If True, include contour logits for training.

        Returns:
            dict with 'onset' and 'frame' logits (batch, n_frames, 127).
            Optionally 'contour' logits (batch, n_contour_frames, n_contour_bins).
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True)

        outputs = self.model(audio)

        result = {
            "onset": outputs["onset"],
            "frame": outputs["note"],
        }
        if return_contour:
            result["contour"] = outputs["contour"]
        return result
