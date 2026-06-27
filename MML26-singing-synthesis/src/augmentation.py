"""Per-file band-limiting augmentation for training audio.

Uses audiomentations' ``BandPassFilter`` and ``LowPassFilter``:
  * https://iver56.github.io/audiomentations/waveform_transforms/band_pass_filter/
  * https://iver56.github.io/audiomentations/waveform_transforms/low_pass_filter/

Per file, the pass-band is tied to the file's *own* vocal range, read from the
frame labels:

  * Top edge  : ALWAYS 2 octaves above the highest active pitch  (f_max * 4).
  * Bottom edge: randomized between 1 and 2 octaves below the lowest active
                 pitch                                          (f_min / 2**U(1,2)).

We realize that band with a ``BandPassFilter`` whose center/bandwidth are pinned
(min == max) to the exact edges we computed for the file, followed by a
``LowPassFilter`` pinned to the same top edge (the ``LowPassFilter`` the user
asked for explicitly; it reinforces the band's upper roll-off).

Files with no active notes are passed through untouched. The augmentation fires
on a fraction ``p`` of samples (default 0.5) so the model still sees plenty of
clean, full-band vocals — applying it to every sample would shift the whole
training distribution toward "filtered" and hurt on full-band test audio.

audiomentations operates on per-sample float32 NumPy arrays, so each batch item
is round-tripped through NumPy.
"""

from __future__ import annotations

import numpy as np
import torch
from audiomentations import BandPassFilter, LowPassFilter

from .constants import SAMPLE_RATE


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _active_pitch_range(frame_labels: torch.Tensor) -> tuple[int, int] | None:
    """Return (min_midi, max_midi) of active pitches in one file, or None.

    ``frame_labels`` is ``(n_frames, n_pitches)`` with non-zero where a pitch is
    active. The pitch axis index is the MIDI note number.
    """
    active_pitches = torch.nonzero(frame_labels.any(dim=0)).flatten()
    if active_pitches.numel() == 0:
        return None
    return int(active_pitches.min()), int(active_pitches.max())


def _pinned_bandpass(low_hz: float, high_hz: float, sample_rate: int) -> BandPassFilter:
    """A BandPassFilter whose center/bandwidth collapse to a fixed [low, high].

    audiomentations parameterizes the band by center frequency and bandwidth
    *fraction* (bandwidth = fraction * center), sampled within [min, max] each
    call. Setting min == max pins the random draw to the exact edges we want.
    Cutoffs are clamped to a safe sub-Nyquist range first.
    """
    nyquist = 0.5 * sample_rate
    low_hz = max(20.0, min(low_hz, nyquist - 100.0))
    high_hz = max(low_hz + 50.0, min(high_hz, nyquist - 50.0))

    center = (low_hz + high_hz) / 2.0
    bandwidth_fraction = (high_hz - low_hz) / center
    bandwidth_fraction = min(max(bandwidth_fraction, 0.01), 1.99)

    return BandPassFilter(
        min_center_freq=center,
        max_center_freq=center,
        min_bandwidth_fraction=bandwidth_fraction,
        max_bandwidth_fraction=bandwidth_fraction,
        p=1.0,
    )


def _pinned_lowpass(high_hz: float, sample_rate: int) -> LowPassFilter:
    nyquist = 0.5 * sample_rate
    high_hz = max(50.0, min(high_hz, nyquist - 50.0))
    return LowPassFilter(
        min_cutoff_freq=high_hz,
        max_cutoff_freq=high_hz,
        p=1.0,
    )


def augment_audio(
    audio: torch.Tensor,
    frame_labels: torch.Tensor,
    *,
    p: float = 0.7,
    sample_rate: int = SAMPLE_RATE,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Band-limit each item in a batch to its own (octave-padded) vocal range.

    Args:
        audio:        ``(B, 1, T)`` or ``(B, T)`` waveform batch.
        frame_labels: ``(B, n_frames, n_pitches)`` frame activity used to read
                      each file's pitch range (MIDI = pitch-axis index).
        p:            probability of filtering each individual sample.
        sample_rate:  audio sample rate in Hz.
        generator:    optional ``torch.Generator`` for reproducibility.

    Returns:
        A new tensor, same shape/device/dtype as ``audio``, with the band-pass +
        low-pass applied to a random ~``p`` fraction of samples and the rest
        passed through unchanged.
    """
    squeezed = audio.dim() == 2
    if squeezed:
        audio = audio.unsqueeze(1)  # (B, 1, T)

    out = audio.clone()
    batch_size = audio.shape[0]

    def _rand() -> float:
        return float(torch.rand(1, generator=generator, device="cpu").item())

    for i in range(batch_size):
        if _rand() >= p:
            continue

        pitch_range = _active_pitch_range(frame_labels[i])
        if pitch_range is None:
            continue
        min_midi, max_midi = pitch_range

        f_min = _midi_to_hz(min_midi)
        f_max = _midi_to_hz(max_midi)

        # Top: always 2 octaves above the highest note.
        high_hz = f_max * 4.0
        # Bottom: randomized 1–2 octaves below the lowest note.
        low_octaves = 1.0 + _rand()  # U(1, 2)
        low_hz = f_min / (2.0 ** low_octaves)

        bandpass = _pinned_bandpass(low_hz, high_hz, sample_rate)
        lowpass = _pinned_lowpass(high_hz, sample_rate)

        samples = audio[i].detach().cpu().numpy().astype(np.float32)
        samples = bandpass(samples=samples, sample_rate=sample_rate)
        samples = lowpass(samples=samples, sample_rate=sample_rate)

        out[i] = torch.from_numpy(samples).to(device=out.device, dtype=out.dtype)

    if squeezed:
        out = out.squeeze(1)
    return out
