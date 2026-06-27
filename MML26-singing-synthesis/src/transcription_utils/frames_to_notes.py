# Code from https://github.com/spotify/basic-pitch/blob/main/basic_pitch/note_creation.py#L347
from typing import Optional, Tuple, List, Literal

import librosa
import numpy as np
import scipy

from scipy.signal import find_peaks
from scipy.stats import mode as scipy_mode  # Use alias to avoid potential conflicts

# cv2 is optional - only needed for adaptive thresholding
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

from ..constants import FRAME_DURATION_S, MIN_NOTE_LEN_FRAMES

def weighted_mode(values: np.ndarray, weights: np.ndarray) -> int:
    """Return the value with the highest total weight."""
    unique_vals = np.unique(values)
    best_val = int(values[0])
    best_weight = -1.0
    for v in unique_vals:
        w = weights[values == v].sum()
        if w > best_weight:
            best_weight = w
            best_val = int(v)
    return best_val


def get_infered_onsets(onsets: np.array, frames: np.array, n_diff: int = 2) -> np.array:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    max_frame_diff = np.max(frame_diff)
    if max_frame_diff > 0:
        frame_diff = (
            np.max(onsets) * frame_diff / max_frame_diff
        )  # rescale to have the same max as onsets

    max_onsets_diff = np.max(
        [onsets, frame_diff], axis=0
    )  # use the max of the predicted onsets and the differences

    return max_onsets_diff


def constrain_frequency(
    onsets: np.array,
    frames: np.array,
    max_freq: Optional[float],
    min_freq: Optional[float],
) -> Tuple[np.array, np.array]:
    """Zero out activations above or below the max/min frequencies

    Args:
        onsets: Onset activation matrix (n_times, n_freqs)
        frames: Frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep.
        min_freq: the minimum frequency to keep.

    Returns:
       The onset and frame activation matrices, with frequencies outside the min and max
       frequency set to 0.
    """
    if max_freq is not None:
        max_freq_idx = int(np.round(librosa.hz_to_midi(max_freq)))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(librosa.hz_to_midi(min_freq)))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0

    return onsets, frames


def output_to_notes_polyphonic(
    frames: np.array,
    onsets: np.array,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: float = librosa.note_to_hz("G9"),
    min_freq: float = librosa.note_to_hz("C1"),
    melodia_trick: bool = False,
    energy_tol: int = 11,
) -> np.array:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        melodia_trick : Whether to use the melodia trick to better detect notes.
        energy_tol: Drop notes below this energy.

    Returns:
        numpy array with the notes of shape [n_notes, (start_time_frames, end_time_frames, pitch_midi)]
    """

    n_frames = frames.shape[0]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if frames.shape[-1] > freq_idx + 1:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # add the note
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx,
            )
        )
    if len(note_events) == 0:
        note_events = np.empty((0, 3))
    else:
        note_events = np.array(note_events).astype(float)
    if note_events.ndim == 1:
        note_events = note_events[np.newaxis, :]
    if note_events.size > 0:
        note_events[:, 0] = note_events[:, 0] * FRAME_DURATION_S
        note_events[:, 1] = note_events[:, 1] * FRAME_DURATION_S

    note_events = np.flip(note_events, 0)  # Put them in direct order

    return note_events


def extract_monophonic_notes(
    onset_probs: np.ndarray,
    vad_probs: np.ndarray,
    pitch_class_probs: np.ndarray,
    octave_probs: np.ndarray,
    onset_threshold: float,
    vad_threshold: float,
    frame_length_s: float,
    n_frames_peak_dominance: int = 3,
    min_note_duration_s: float = 0.07,
    onset_threhold_type: Literal[
        "fixed", "adaptive-mean", "adaptive-gaussian", "adaptive-otsu"
    ] = "fixed",
    pitch_vote: Literal["mode", "weighted"] = "mode",
) -> np.ndarray:
    """
    Extracts monophonic notes from frame-wise predictions.

    Identifies onsets as local peaks in onset probabilities above a threshold,
    ensuring the peak is the maximum within a specified frame window.
    Determines offsets based on VAD drops or subsequent onsets.
    Calculates the pitch as the mode of predicted MIDI notes between onset and offset.
    Filters out notes shorter than a minimum duration.

    Args:
        onset_probs (np.ndarray): Array of shape (n_frames,) with onset probabilities.
        vad_probs (np.ndarray): Array of shape (n_frames,) with voice activity probabilities.
        pitch_class_probs (np.ndarray): Array of shape (n_frames, 12) with pitch class probabilities.
        octave_probs (np.ndarray): Array of shape (n_frames, 8) with octave probabilities.
        onset_threshold (float): Threshold for detecting onset peaks.
        vad_threshold (float): Threshold for detecting voice activity offset.
        frame_length_s (float): Duration of each frame in seconds.
        n_frames_peak_dominance (int, optional): An onset peak must be the maximum value
            within a centered window of this many frames. Defaults to 1 (peak > neighbors).
            Must be an odd positive integer for centered window logic. If even, it's treated
            as the next odd number up for window calculation.
        min_note_duration_s (float, optional): Minimum duration (in seconds) for a note
            to be included in the output. Defaults to 0.0.
        pitch_vote: Pitch aggregation method. "mode" uses the plain
            most-frequent pitch (unweighted). "weighted" applies a
            Hanning window so that centre frames count more than edges,
            reducing the influence of unstable onset/offset frames.
            Defaults to "mode".

    Returns:
        np.ndarray: Array of shape (n_notes, 3) where each row contains
                    [onset_time_s, offset_time_s, midi_pitch].
                    Returns an empty array shape (0, 3) if no notes are found.
    """
    # --- Input Validation ---
    if not (
        onset_probs.ndim == 1
        and vad_probs.ndim == 1
        and pitch_class_probs.ndim == 2
        and pitch_class_probs.shape[1] == 12
        and octave_probs.ndim == 2
        and octave_probs.shape[1] == 8
        and onset_probs.shape[0]
        == vad_probs.shape[0]
        == pitch_class_probs.shape[0]
        == octave_probs.shape[0]
    ):
        raise ValueError("Input array dimensions are incorrect.")
    if not isinstance(n_frames_peak_dominance, int) or n_frames_peak_dominance < 1:
        raise ValueError("n_frames_peak_dominance must be a positive integer.")
    if not isinstance(min_note_duration_s, (float, int)) or min_note_duration_s < 0:
        raise ValueError("min_note_duration_s must be a non-negative number.")

    n_frames = onset_probs.shape[0]
    if n_frames == 0:
        return np.empty((0, 3), dtype=float)

    # --- 1. Identify Onset Frame Indices ---
    # Use find_peaks' distance parameter for dominance filtering:
    # among peaks within `distance` of each other, only the tallest is kept.
    half_window = (
        (n_frames_peak_dominance - 1) // 2
        if n_frames_peak_dominance % 2 != 0
        else n_frames_peak_dominance // 2
    )
    peak_distance = max(1, half_window + 1)

    if onset_threhold_type == "fixed":
        onset_indices, _ = find_peaks(onset_probs, height=onset_threshold, distance=peak_distance)
    elif onset_threhold_type == "adaptive-otsu":
        if not HAS_CV2:
            raise ImportError("opencv-python (cv2) is required for adaptive-otsu thresholding. Install with: pip install opencv-python")
        cv2_onset_probs = (onset_probs * 255).astype(np.uint8)[:, np.newaxis]
        _, onset_probs_mask = cv2.threshold(
            cv2_onset_probs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        onset_probs_mask = (onset_probs_mask / 255).squeeze(-1)
        onset_probs = onset_probs * onset_probs_mask
        onset_indices, _ = find_peaks(onset_probs, height=0.01, distance=peak_distance)
    elif onset_threhold_type == "adaptive-mean":
        if not HAS_CV2:
            raise ImportError("opencv-python (cv2) is required for adaptive-mean thresholding. Install with: pip install opencv-python")
        block_size = 101
        cv2_onset_probs = (onset_probs * 255).astype(np.uint8)[:, np.newaxis]
        onset_probs_mask = cv2.adaptiveThreshold(
            cv2_onset_probs,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            6,
        )
        onset_probs_mask = (onset_probs_mask / 255).squeeze(-1)
        onset_probs = onset_probs * onset_probs_mask
        onset_indices, _ = find_peaks(onset_probs, height=0.01, distance=peak_distance)
    elif onset_threhold_type == "adaptive-gaussian":
        if not HAS_CV2:
            raise ImportError("opencv-python (cv2) is required for adaptive-gaussian thresholding. Install with: pip install opencv-python")
        block_size = 101
        cv2_onset_probs = (onset_probs * 255).astype(np.uint8)[:, np.newaxis]
        onset_probs_mask = cv2.adaptiveThreshold(
            cv2_onset_probs,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            6,
        )
        onset_probs_mask = (onset_probs_mask / 255).squeeze(-1)
        onset_probs = onset_probs * onset_probs_mask
        onset_indices, _ = find_peaks(onset_probs, height=0.01, distance=peak_distance)

    if len(onset_indices) == 0:
        return np.empty((0, 3), dtype=float)  # No onsets found

    # --- Pre-process VAD for adaptive thresholding (done once, not per-note) ---
    if onset_threhold_type != "fixed":
        if not HAS_CV2:
            raise ImportError(
                "opencv-python (cv2) is required for adaptive thresholding. "
                "Install with: pip install opencv-python"
            )
        cv2_vad_probs = (vad_probs * 255).astype(np.uint8)[:, np.newaxis]
        if onset_threhold_type == "adaptive-otsu":
            _, vad_probs_mask = cv2.threshold(
                cv2_vad_probs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif onset_threhold_type == "adaptive-mean":
            vad_probs_mask = cv2.adaptiveThreshold(
                cv2_vad_probs, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 101, 6,
            )
        elif onset_threhold_type == "adaptive-gaussian":
            vad_probs_mask = cv2.adaptiveThreshold(
                cv2_vad_probs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 101, 6,
            )
        vad_probs = (vad_probs_mask / 255).squeeze(-1)

    # --- Precompute all VAD-below-threshold frame indices once (O(T)) ---
    # Then use searchsorted per note to find the first offset after each onset (O(log T))
    vad_below_threshold = np.where(vad_probs < vad_threshold)[0]

    # --- Main Note Extraction Loop ---
    notes = []
    current_frame = 0
    onset_idx_ptr = 0  # Pointer to the current onset index we are considering

    while onset_idx_ptr < len(onset_indices):
        # Find the next onset index at or after current_frame
        while (
            onset_idx_ptr < len(onset_indices)
            and onset_indices[onset_idx_ptr] < current_frame
        ):
            onset_idx_ptr += 1

        if onset_idx_ptr >= len(onset_indices):
            break  # No more onsets found after current_frame

        onset_idx = onset_indices[onset_idx_ptr]

        # --- 2. Identify offset frame index ---
        offset_idx = n_frames  # Default offset is end of sequence

        # a) Find first VAD drop below threshold after onset via binary search
        search_pos = np.searchsorted(vad_below_threshold, onset_idx + 1, side='left')
        if search_pos < len(vad_below_threshold):
            vad_offset_idx = vad_below_threshold[search_pos]
            offset_idx = min(offset_idx, vad_offset_idx)

        # b) Find the frame before the *next* onset
        if onset_idx_ptr + 1 < len(onset_indices):
            next_onset_idx = onset_indices[onset_idx_ptr + 1]
            offset_idx = min(offset_idx, next_onset_idx)

        # Ensure offset is strictly after onset
        if offset_idx <= onset_idx:
            # This onset is too short or immediately followed by silence/next note. Skip it.
            # Advance current_frame past this onset to avoid infinite loops if conditions persist.
            current_frame = onset_idx + 1
            onset_idx_ptr += 1  # Move to consider the next potential onset
            continue

        # --- 3. Check Minimum Duration ---
        onset_time = onset_idx * frame_length_s
        offset_time = offset_idx * frame_length_s
        duration_s = offset_time - onset_time

        if duration_s < min_note_duration_s:
            # Note is too short, skip it, but advance state
            current_frame = offset_idx
            onset_idx_ptr += 1  # Ensure we always consider the next onset index
            continue

        # --- 4. Extract pitch between onset (inclusive) and offset (exclusive) ---
        segment_start = onset_idx
        segment_end = offset_idx
        pitch = -1  # Default invalid pitch
        segment_midi_pitches = np.array([], dtype=int)

        if segment_start < segment_end:  # Ensure the segment has duration
            # Get the most likely pitch class and octave for each frame in the segment
            segment_pitch_classes = np.argmax(
                pitch_class_probs[segment_start:segment_end], axis=1
            )
            segment_octaves = np.argmax(octave_probs[segment_start:segment_end], axis=1)

            # Calculate MIDI pitch for each frame (assuming octave 0 starts at MIDI 0)
            # Adjust if your octave definition maps differently to MIDI numbers
            segment_midi_pitches = segment_octaves * 12 + segment_pitch_classes

            # Determine pitch from frame-level MIDI predictions
            if len(segment_midi_pitches) > 0:
                if pitch_vote == "weighted":
                    w = np.hanning(len(segment_midi_pitches))
                    if w.sum() == 0:  # len <= 2: hanning is all zeros
                        w = np.ones(len(segment_midi_pitches))
                    pitch = weighted_mode(segment_midi_pitches, w)
                else:
                    result = scipy_mode(segment_midi_pitches)
                    pitch = result.mode
            # else: pitch remains -1 (segment was empty for pitch calculation)
        if pitch < librosa.note_to_midi("C1"):
            pitch = -1  # pitch not valid
        if pitch > librosa.note_to_midi("C8"):
            pitch = -1  # pitch not valid

        # --- 5. Store Note ---
        # Check if a valid pitch was found (optional, could store notes with pitch=-1)
        if pitch != -1:
            notes.append([onset_time, offset_time, pitch])

        # --- Update State ---
        # Update current_frame to the determined offset to search for the next note
        current_frame = offset_idx
        # Ensure we always consider the next onset index in the list
        onset_idx_ptr += 1

    if len(notes) == 0:
        return np.empty((0, 3), dtype=float)

    return np.array(notes, dtype=float)
