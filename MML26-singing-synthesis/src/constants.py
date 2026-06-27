import math

# Audio / frame grid (16 kHz, hop 256 → 62.5 fps)
SAMPLE_RATE: int = 16_000
HOP_LENGTH: int = 256
FEATURE_RATE: float = SAMPLE_RATE / HOP_LENGTH  # 62.5 frames per second
FRAME_DURATION_S: float = HOP_LENGTH / SAMPLE_RATE  # 16 ms per frame

# ~60 ms minimum note length (matches prior 3-frame threshold at 50 fps / 20 ms frames)
MIN_NOTE_LEN_FRAMES: int = max(3, round(0.060 / FRAME_DURATION_S))

ACTIVE_FRAME_LABEL = 2
ONSET_LABEL = 3

N_PITCHES: int = 127
OAF_NUM_LABELS: int = N_PITCHES * 2  # onset + frame per pitch
PIANO_MIDI_START: int = 21  # A0; Basic Pitch outputs 88 piano keys from this MIDI pitch

# Basic Pitch architecture
BASIC_PITCH_N_SEMITONES: int = 88
BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE: int = 3
BASIC_PITCH_BASE_FREQUENCY: float = 27.5
BASIC_PITCH_MAX_N_SEMITONES: int = int(
    math.floor(
        12.0 * math.log2(0.5 * SAMPLE_RATE / BASIC_PITCH_BASE_FREQUENCY)
    )
)
BASIC_PITCH_DEFAULT_HARMONICS = [0.5, 1, 2, 3, 4, 5, 6, 7]
BASIC_PITCH_N_CONTOUR_BINS = (
    BASIC_PITCH_N_SEMITONES * BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE
)

DEBUG_PRINT_FRAME_AND_ONSET_DURING_EVALUATION = True

WANDB_PROJECT = "basic_pitch+_MML-hackaton2026"
