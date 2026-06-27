# Project Context: Singing Voice Transcription with Synthetic Data

> This file is the working context for the project **as it currently stands**. It
> reflects what is actually implemented in the repo — not the original plan. See
> `README.md` for the full, detailed write-up.

## 1. Objective
Build a robust **Singing Voice Transcription (SVT)** model that predicts the
**onset**, **offset**, and **pitch** of lead vocals. Manually annotating commercial
music is expensive, so we instead **synthesize singing** from existing note
annotations (`scores/*.tsv`) and train the provided **Basic Pitch** CNN on it.
Challenge repo: `Klangio/MML26-singing-synthesis`.

## 2. Reference model & hard constraints (verified against the code)
- **Model:** a slightly modified **Basic Pitch** CNN (`src/basic_pitch_model.py`).
- **Sample rate is 16 kHz**, not 22050. `src/constants.py: SAMPLE_RATE = 16000`. The
  dataloader (`src/dataloading/base_dataloader.py::_load_audio`) **auto-resamples any
  wav to 16 kHz**, so output SR is flexible. We write **16 kHz mono int16** WAVs.
- **Data contract:** the dataloader iterates `syntheticdataset/<id>/` for
  `audio.wav` + `score.tsv`. We feed data by writing that layout; we do **not** modify
  the training/model code (challenge rule).
- **`score.tsv` MUST start with a header line.** The loader reads labels with
  `np.loadtxt(..., skiprows=1)`, so a missing header silently drops the first note. We
  prepend `# onset,offset,note`.
- **Pitch range:** the note decoder is valid for **C1–C8 (MIDI 24–108)**; the CQT
  front-end spans A0–C8 (anchored at A0 = 27.5 Hz). Our source scores are **MIDI 29–83**,
  comfortably in range. MIDI→Hz uses the standard `f = 440·2**((midi−69)/12)`, which
  matches the model's CQT — so our pitch convention is identical to the model's.

## 3. CRITICAL CONSTRAINT
**Never alter the fundamental pitch or timing of the generated audio unless we also
rewrite the `.tsv`.** `score.tsv` is the ground truth the model is graded against, so
audio and labels must stay aligned by construction.

## 4. The pipeline — what we actually have

### Phase A — Method 2 only: Speech-to-Singing (V2S via the WORLD vocoder)
`scripts/A/vocal_synth_method_2.py`. Borrow a **LibriTTS** speaker's *timbre* and impose
the score's *melody* on it:
1. Decompose a speaker's speech with WORLD → **F0**, **spectral envelope (`sp`)**,
   **aperiodicity (`ap`)**.
2. Discard the speaker's F0; build a new **F0 contour from the TSV MIDI** (per-note Hz,
   silence in the gaps).
3. Resynthesize (`sp` + `ap` + our F0) → a dry, pitch-exact "singer". Mute inter-note
   gaps, peak-normalize, write **16 kHz int16** `audio.wav` + headered `score.tsv`.

Pitch is **exact by construction** (the only pitched layer is the F0 we author from the
TSV). Different speakers → different timbres at identical pitch/timing (the Phase A goal).
Defaults: **3 speakers/score**, **f0-matched** speaker selection (octave-folded distance
to the score's median pitch), LibriTTS `dev-clean` auto-downloaded. Distributed via
`--shard I/K` (deterministic per-score speaker choice; `--skip-existing` to resume).

> **We do NOT have Method 1** (`midi2voice`) and **do NOT have Phase B** (accompaniment /
> instrumental generation). Those remain future work.

### Phase C — On-the-fly band-limiting augmentation (training time)
`src/augmentation.py` (`augment_audio`), called inside `src/lightning_module.py`
`training_step` (the hook the challenge owner left). Per training file, it band-limits
the audio to that file's **own vocal range**, read from the frame labels:
- top edge = **2 octaves above** the highest active note (`f_max*4`);
- bottom edge = **1–2 octaves below** the lowest active note (`f_min / 2**U(1,2)`);
- realized as a pinned `BandPassFilter` + `LowPassFilter` (audiomentations).
Applied to a fraction **p = 0.6** of samples; files with no notes pass through. It is
**label-preserving** (band-pass/low-pass shift neither time nor pitch) and **training-only**
(validation/test/the hidden set see raw audio). This is a lighter, on-the-fly stand-in for
the originally-planned offline Phase C (reverb/EQ/compression/SIR mixing), which is **not**
implemented.

### Phase D — Training
`MML26-singing-synthesis/experiments_sample.sh` → `python3 -m src.train`. Trains on
`Synthetic` (our data), validates on `Klangio`. Key settings: 8 s chunks, batch 32,
`frame-weight 9`, `onset-weight 18`, lr 1e-4, eval metric **COnPOff_f1**, precision 32,
**`--limit-train-batches 200`** (cap epoch size), `--logger wandb`. Early stopping
(patience 10) + best-`COnPOff_f1` checkpoint are built into `train.py`.

## 5. Papers / anchors (and their role)
- **WORLD vocoder** — Morise et al., 2016. The DSP engine that decomposes speech into
  F0 / spectral envelope / aperiodicity and resynthesizes. *Role:* makes pitch and timbre
  independent, so swapping F0 leaves formants untouched.
- **V2S (Voice-to-Singing)** — arXiv:2102.08575, Basak et al., 2021. Establishes that
  imposing an F0 contour over arbitrary speech (no phoneme→note alignment) yields usable
  "singing" for training. *Role:* validates Method 2's mechanism. **Our adaptation:** the
  paper borrows F0 from a real singing recording; we instead author F0 from the TSV, so
  pitch is exact and label-aligned. (We deliberately did *not* use Saitou 2007 — it models
  F0 micro-dynamics we don't need.)
- **Basic Pitch** — Bittner et al. (Spotify), ICASSP 2022. The transcription model we
  train. *Role:* defines the targets (onset/frame/contour), the valid pitch range, and the
  16 kHz input.

## 6. File inventory
- `scripts/A/vocal_synth_method_2.py` — Phase A Method 2 synthesizer (sharding, f0-match,
  self-check). Usage commands are documented at the top of the file.
- `MML26-singing-synthesis/src/augmentation.py` — Phase C band-limiting augmentation.
- `MML26-singing-synthesis/src/lightning_module.py` — `training_step` calls `augment_audio`.
- `MML26-singing-synthesis/experiments_sample.sh` — training launcher.
- `MML26-singing-synthesis/requirements.txt` — adds `pyworld`, `audiomentations`.

## 7. Future work (not implemented)
Method 1 (`midi2voice`/RVC), Phase B accompaniment, full offline Phase C
(reverb/EQ/compression + SIR mixing with instrumentals → true implicit source separation),
and a standalone `dataset_loader.py`.
