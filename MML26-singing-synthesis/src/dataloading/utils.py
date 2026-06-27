from pathlib import Path
from typing import Union

import numpy as np
import torch


def notes_from_midi(midi: Union[str, Path]) -> np.ndarray:
    """Extract notes from a MIDI file as a numpy array.

    Args:
        midi: Path to the MIDI file.

    Returns:
        Array of shape (N, 3) with columns [onset_s, offset_s, midi_pitch].
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi))
    notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append([note.start, note.end, note.pitch])
    return np.array(notes)


def stack_dicts(dict_list: list) -> dict:
    """Stack a list of dicts with the same keys into a single dict.

    Handles tensors, strings, lists of strings, and booleans.

    Args:
        dict_list: List of dictionaries with the same keys.

    Returns:
        Collated dictionary; tensors are concatenated along dim 0,
        all other types are collected into lists.
    """
    collated_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in collated_dict:
                collated_dict[key] = []
            if isinstance(value, (torch.Tensor, str, bool, list)):
                collated_dict[key].append(value)
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    for key, value_list in collated_dict.items():
        if isinstance(value_list[0], torch.Tensor):
            collated_dict[key] = torch.concat(value_list, 0)

    return collated_dict
