from typing import Dict

import librosa
import numpy as np
from mir_eval.transcription import (
    onset_precision_recall_f1,
    precision_recall_f1_overlap,
)
from mir_eval.util import match_events


METRICS_REGISTRY = {
        "COn_precision": 0,
        "COn_recall": 0,
        "COn_f1": 0,
        "COnP_precision": 0,
        "COnP_recall": 0,
        "COnP_f1": 0,
        "COnOff_precision": 0,
        "COnOff_recall": 0,
        "COnOff_f1": 0,
        "COnPOff_precision": 0,
        "COnPOff_recall": 0,
        "COnPOff_f1": 0,
        "pitch_mse": np.inf,  # Infinite error when no pitch
        # Pitch class metrics
        "COnPC_precision": 0,
        "COnPC_recall": 0,
        "COnPC_f1": 0,
        "COnPCOff_precision": 0,
        "COnPCOff_recall": 0,
        "COnPCOff_f1": 0,
}


def get_metrics_dict(
    ref_notes: np.array, est_notes: np.array, onset_tolerance: float = 0.05
) -> Dict[str, float]:
    """
    Get comprehensive metrics for singing voice transcription evaluation.
    
    Metrics include:
    - COn: Onset-only matching
    - COnP: Onset + Pitch matching
    - COnOff: Onset + Offset matching (pitch-agnostic)
    - COnPOff: Onset + Pitch + Offset matching
    - pitch_mse: Mean Squared Error of MIDI pitch for onset-matched notes
    - COnPC: Onset + Pitch Class matching (octave-invariant)
    - COnPCOff: Onset + Pitch Class + Offset matching (octave-invariant)
    
    :param ref_notes: Reference notes with shape [n_notes, 3] (3 -> (onset, offset, pitch))
    :param est_notes: Estimated notes with shape [n_notes, 3] (3 -> (onset, offset, pitch))
    :param onset_tolerance: Onset difference tolerance. 50ms default.
    :return: Dictionary with all evaluation metrics.
    """
    assert len(ref_notes) > 0
    metrics = METRICS_REGISTRY.copy()
    
    # Ensure inputs are numpy arrays
    ref_notes = np.array(ref_notes)
    est_notes = np.array(est_notes)

    if est_notes.size == 0:  # No note predicted
        return metrics

    ref_intervals = ref_notes[:, :-1]
    est_intervals = est_notes[:, :-1]
    ref_pitches = librosa.midi_to_hz(ref_notes[:, -1])
    est_pitches = librosa.midi_to_hz(est_notes[:, -1])

    onset_precision, onset_recall, onset_f_measure = onset_precision_recall_f1(
        ref_intervals, est_intervals, onset_tolerance=onset_tolerance
    )
    metrics["COn_precision"] = onset_precision
    metrics["COn_recall"] = onset_recall
    metrics["COn_f1"] = onset_f_measure

    precision, recall, f_measure, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None,
        offset_min_tolerance=0.05,
    )

    metrics["COnP_precision"] = precision
    metrics["COnP_recall"] = recall
    metrics["COnP_f1"] = f_measure

    # COnOff: Onset and Offset matching (ignoring pitch)
    # Use a very large pitch tolerance to effectively ignore pitch differences
    precision, recall, f_measure, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=np.inf,  # Ignore pitch differences
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )

    metrics["COnOff_precision"] = precision
    metrics["COnOff_recall"] = recall
    metrics["COnOff_f1"] = f_measure

    # COnPOff: Onset, Pitch and Offset matching
    precision, recall, f_measure, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )

    metrics["COnPOff_precision"] = precision
    metrics["COnPOff_recall"] = recall
    metrics["COnPOff_f1"] = f_measure
    
    # Pitch MSE: Mean Squared Error for matched notes
    # Match notes based on onset
    ref_onsets = ref_intervals[:, 0]
    est_onsets = est_intervals[:, 0]
    matching = np.array(match_events(ref_onsets, est_onsets, onset_tolerance))
    
    if len(matching) > 0:
        # For matched notes, compute squared error in MIDI pitch
        ref_matched_pitches = ref_notes[matching[:, 0], -1]
        est_matched_pitches = est_notes[matching[:, 1], -1]
        pitch_errors = ref_matched_pitches - est_matched_pitches
        metrics["pitch_mse"] = np.mean(pitch_errors ** 2)
    else:
        metrics["pitch_mse"] = np.inf  # No matches
    
    # Pitch Class Metrics: Same as above but using pitch class instead of absolute pitch
    # Convert MIDI pitches to pitch classes (mod 12)
    ref_pitch_classes = ref_notes[:, -1] % 12
    est_pitch_classes = est_notes[:, -1] % 12
    
    # Map pitch classes to pseudo-frequencies for mir_eval
    # We'll map pitch class to a frequency in a single octave (C4=60)
    ref_pc_midi = 60 + ref_pitch_classes
    est_pc_midi = 60 + est_pitch_classes
    ref_pc_hz = librosa.midi_to_hz(ref_pc_midi)
    est_pc_hz = librosa.midi_to_hz(est_pc_midi)
    
    # COnPC: Onset + Pitch Class (no offset)
    precision, recall, f_measure, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pc_hz,
        est_intervals,
        est_pc_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,  # ~1 semitone tolerance for pitch class
        offset_ratio=None,
        offset_min_tolerance=0.05,
    )
    
    metrics["COnPC_precision"] = precision
    metrics["COnPC_recall"] = recall
    metrics["COnPC_f1"] = f_measure
    
    # COnPCOff: Onset + Pitch Class + Offset
    precision, recall, f_measure, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pc_hz,
        est_intervals,
        est_pc_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,  # ~1 semitone tolerance for pitch class
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )
    
    metrics["COnPCOff_precision"] = precision
    metrics["COnPCOff_recall"] = recall
    metrics["COnPCOff_f1"] = f_measure
    
    return metrics
