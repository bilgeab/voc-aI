"""
Basic Pitch CNN architecture (Spotify ICASSP 2022), implemented in PyTorch.
Adapted from https://github.com/gudgud96/basic-pitch-torch
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from nnAudio.features import CQT2010v2
from torch import Tensor, nn

from .constants import (
    BASIC_PITCH_BASE_FREQUENCY,
    BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
    BASIC_PITCH_DEFAULT_HARMONICS,
    BASIC_PITCH_MAX_N_SEMITONES,
    BASIC_PITCH_N_SEMITONES,
    HOP_LENGTH,
    N_PITCHES,
    PIANO_MIDI_START,
    SAMPLE_RATE,
)


def log_base_b(x: Tensor, base: int) -> Tensor:
    numerator = torch.log(x)
    denominator = torch.log(
        torch.tensor([base], dtype=numerator.dtype, device=numerator.device)
    )
    return numerator / denominator


def normalized_log(inputs: Tensor) -> Tensor:
    power = torch.square(inputs)
    log_power = 10 * log_base_b(power + 1e-10, 10)

    log_power_min = torch.amin(log_power, dim=(1, 2)).reshape(inputs.shape[0], 1, 1)
    log_power_offset = log_power - log_power_min
    log_power_offset_max = torch.amax(log_power_offset, dim=(1, 2)).reshape(
        inputs.shape[0], 1, 1
    )
    log_power_normalized = log_power_offset / log_power_offset_max
    log_power_normalized = torch.nan_to_num(log_power_normalized, nan=0.0)

    return log_power_normalized.reshape(inputs.shape)


def label_frames_from_samples(n_samples: int) -> int:
    return (n_samples - 1) // HOP_LENGTH + 1


def align_to_label_frames(tensor: Tensor, n_samples: int) -> Tensor:
    """Align a (batch, time, …) tensor to the label frame grid."""
    target_frames = label_frames_from_samples(n_samples)
    current_frames = tensor.shape[1]
    if current_frames == target_frames:
        return tensor
    if current_frames == target_frames + 1:
        return tensor[:, :-1, ...]
    if current_frames > target_frames:
        return tensor[:, :target_frames, ...]
    pad_frames = target_frames - current_frames
    return F.pad(tensor, (0, 0, 0, pad_frames))


def piano_logits_to_midi_layout(piano_logits: Tensor) -> Tensor:
    """Pad 88-key piano logits into the 127-bin MIDI layout (bins 0–20 are zero)."""
    out = piano_logits.new_zeros(
        piano_logits.shape[0],
        piano_logits.shape[1],
        N_PITCHES,
    )
    piano_end = PIANO_MIDI_START + BASIC_PITCH_N_SEMITONES
    out[:, :, PIANO_MIDI_START:piano_end] = piano_logits
    return out


class HarmonicStacking(nn.Module):
    def __init__(
        self,
        bins_per_semitone: int,
        harmonics: List[float],
        n_output_freqs: int,
    ):
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs
        self.shifts = [
            int(round(12.0 * self.bins_per_semitone * math.log2(h)))
            for h in self.harmonics
        ]

    def forward(self, x: Tensor) -> Tensor:
        hcqt = []
        for shift in self.shifts:
            if shift == 0:
                cur_cqt = x
            elif shift > 0:
                cur_cqt = F.pad(x[:, :, shift:], (0, shift))
            else:
                cur_cqt = F.pad(x[:, :, :shift], (-shift, 0))
            hcqt.append(cur_cqt)
        hcqt = torch.stack(hcqt, dim=1)
        return hcqt[:, :, :, : self.n_output_freqs]


class BasicPitchTorch(nn.Module):
    def __init__(
        self,
        stack_harmonics: List[float] | None = None,
    ) -> None:
        super().__init__()
        if stack_harmonics is None:
            stack_harmonics = list(BASIC_PITCH_DEFAULT_HARMONICS)
        self.stack_harmonics = stack_harmonics
        if stack_harmonics:
            self.hs = HarmonicStacking(
                bins_per_semitone=BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
                harmonics=stack_harmonics,
                n_output_freqs=BASIC_PITCH_N_SEMITONES
                * BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
            )
            num_in_channels = len(stack_harmonics)
        else:
            num_in_channels = 1

        n_semitones = int(
            np.min(
                [
                    int(
                        np.ceil(12.0 * np.log2(len(stack_harmonics)))
                        + BASIC_PITCH_N_SEMITONES
                    ),
                    BASIC_PITCH_MAX_N_SEMITONES,
                ]
            )
        )
        self.cqt_layer = CQT2010v2(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=BASIC_PITCH_BASE_FREQUENCY,
            n_bins=n_semitones * BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
            bins_per_octave=12 * BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
            verbose=False,
        )

        CONV_CONTOUR_CHANNELS = 8  # Original model 8
        CONV_NOTE_CHANNELS = 32  # Original model 32
        CONV_ONSET_PRE = 32  # Original model 32
        DROPOUT = 0.0

        self.bn_layer = nn.BatchNorm2d(1, eps=0.001)
        self.conv_contour = nn.Sequential(
            nn.Conv2d(num_in_channels, CONV_CONTOUR_CHANNELS, kernel_size=(3, 3 * 13), padding="same"),
            nn.BatchNorm2d(CONV_CONTOUR_CHANNELS, eps=0.001),
            nn.ReLU(),
            nn.Conv2d(CONV_CONTOUR_CHANNELS, 1, kernel_size=5, padding="same"),
        )
        self.conv_note = nn.Sequential(
            nn.Conv2d(1, CONV_NOTE_CHANNELS, kernel_size=7, stride=(1, 3)),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv2d(CONV_NOTE_CHANNELS, 1, kernel_size=(7, 3), padding="same"),
        )
        self.conv_onset_pre = nn.Sequential(
            nn.Conv2d(num_in_channels, CONV_ONSET_PRE, kernel_size=5, stride=(1, 3)),
            nn.BatchNorm2d(CONV_ONSET_PRE, eps=0.001),
            nn.ReLU(),
        )
        self.conv_onset_post = nn.Sequential(
            nn.Conv2d(CONV_ONSET_PRE + 1, 1, kernel_size=3, stride=1, padding="same"),
        )

    def _compute_cqt(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3 and inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        x = self.cqt_layer(inputs)
        x = torch.transpose(x, 1, 2)
        x = normalized_log(x)
        x = x.unsqueeze(1)
        x = self.bn_layer(x)
        return x.squeeze(1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.dim() == 3 and x.shape[1] == 1:
            n_samples = x.shape[-1]
            x = x.squeeze(1)
        elif x.dim() == 2:
            n_samples = x.shape[-1]
        else:
            raise ValueError(f"Expected audio shape (batch, time) or (batch, 1, time), got {x.shape}")

        cqt = self._compute_cqt(x)
        cqt = align_to_label_frames(cqt, n_samples)
        if hasattr(self, "hs"):
            cqt = self.hs(cqt)
        else:
            cqt = cqt.unsqueeze(1)

        x_contour = self.conv_contour(cqt)
        x_contour_for_note = F.pad(x_contour, (2, 2, 3, 3))
        x_note = self.conv_note(x_contour_for_note)

        cqt_for_onset = F.pad(cqt, (1, 1, 2, 2))
        x_onset_pre = self.conv_onset_pre(cqt_for_onset)
        x_onset_pre = torch.cat([x_note, x_onset_pre], dim=1)
        x_onset = self.conv_onset_post(x_onset_pre)

        contour = align_to_label_frames(x_contour.squeeze(1), n_samples)
        note = align_to_label_frames(
            piano_logits_to_midi_layout(x_note.squeeze(1)), n_samples
        )
        onset = align_to_label_frames(
            piano_logits_to_midi_layout(x_onset.squeeze(1)), n_samples
        )

        return {
            "onset": onset,
            "contour": contour,
            "note": note,
        }
