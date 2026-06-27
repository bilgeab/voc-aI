import numpy as np
import matplotlib.pyplot as plt

def notes_to_pianoroll(note_list, resolution=50, num_pitches=128):
    """
    Convert a list of Mx3 note arrays into a stacked pianoroll plot.

    Args:
        note_list (list of np.ndarray): A list where each entry is an Mx3 note array.
            Note times are in seconds (onset_s, offset_s, midi_pitch).
        resolution (int): Time resolution in milliseconds (default: 50ms).
        num_pitches (int): Number of MIDI pitches (default: 128).

    Returns:
        matplotlib.figure.Figure: The generated stacked pianoroll plot.
    """
    resolution_s = resolution / 1000.0  # Convert ms to seconds for comparison with note times

    if not note_list:  # Handle empty list
        max_time = 0
    else:
        max_time = max((np.max(notes[:, 1]) if len(notes) > 0 else 0) for notes in note_list)

    num_time_bins = int(np.ceil(max_time / resolution_s))
    num_tracks = len(note_list)

    fig, axes = plt.subplots(num_tracks, 1, figsize=(12, 3 * num_tracks), sharex=True)

    if num_tracks == 1:
        axes = [axes]  # Ensure axes is iterable if only one track

    for idx, notes in enumerate(note_list):
        pianoroll = np.zeros((num_pitches, num_time_bins), dtype=int)

        if notes.shape[0] > 0:  # Avoid indexing error if empty
            for start, end, pitch in notes:
                start_idx = int(start / resolution_s)
                end_idx = int(end / resolution_s)
                pianoroll[int(pitch), start_idx:end_idx] = 1

        ax = axes[idx]
        ax.imshow(pianoroll[::-1], aspect="auto", cmap="gray_r", origin="lower")
        ax.set_ylabel(f"Track {idx+1} (Pitch)")
        ax.set_yticks(np.arange(0, num_pitches, 12))
        ax.set_yticklabels(np.arange(num_pitches, 0, -12))

    # X-axis settings (time)
    axes[-1].set_xlabel("Time (seconds)")
    time_ticks = np.linspace(0, num_time_bins, 10, dtype=int)
    time_labels = (time_ticks * resolution) / 1000
    axes[-1].set_xticks(time_ticks)
    axes[-1].set_xticklabels(time_labels)
    return fig
