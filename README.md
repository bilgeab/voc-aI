# voc-aI — Singing Voice Transcription from Synthetic Data

Training a **Singing Voice Transcription (SVT)** model — one that reads a vocal recording
and writes out every note's **onset**, **offset**, and **pitch** — without paying for a
single hand-labelled commercial song. We *generate* the training data: we take note
annotations we already have, synthesize realistic singing whose pitch and timing match
those annotations **exactly**, and train the provided **Basic Pitch** model on it.

This README explains the whole thing end to end: the problem, why it's hard, the research
the approach is anchored on, the synthesis pipeline, the training-time augmentation, and
exactly how to run everything.

---

## 1. The problem

We want a model that transcribes lead vocals into notes. The reference challenge
(`Klangio/MML26-singing-synthesis`) provides:

- a (slightly modified) **Basic Pitch** CNN and a full training/eval pipeline,
- **`scores/*.tsv`** — minimal note annotations for real singing (each line is
  `onset_seconds, offset_seconds, MIDI_pitch`),
- a **validation set** (`klangiodataset/`) and a **hidden test set**.

The catch: we are *not* given the audio for those scores. Hand-annotating commercial music
is expensive, so the challenge is essentially: **can you synthesize singing from the note
annotations, train on it, and match a model trained on real singing?**

The model is scored — higher is better — on three note-level F1 metrics, in order of
importance:

| Metric | Meaning |
|---|---|
| `COnPOff_f1` | Correct **on**set, **p**itch, **off**set |
| `COnP_f1` | Correct **on**set and **p**itch |
| `COn_f1` | Correct **on**set |

Guidance from the challenge: `COnP_f1` should clear **0.15**, or something is wrong with
the data. Note-matching tolerance is **50 ms**.

### Why it's hard
1. **Labels must stay exact.** The model is graded against `score.tsv`. If our synthesized
   audio drifts in pitch or timing even slightly, we are training on wrong labels.
2. **Domain gap.** The hidden test is real singing; naive synthetic audio can be too clean
   or too artificial.
3. **Don't touch the model.** Challenge rules forbid modifying the provided training/model
   code. We can only *write data* into the expected layout and fill the augmentation hook
   the challenge owner deliberately left open.

---

## 2. The core idea

Instead of recording singers, we **build singing whose pitch is correct by construction.**
The trick (Method 2 below) is to take a real person's *voice texture* and force it to follow
a melody we author directly from the TSV. Because that melody is the very thing the model is
graded on, there is no separate "pitch knob" that can drift out of sync.

> **THE CRITICAL CONSTRAINT, everywhere:** never change the fundamental pitch or timing of
> the audio unless we also rewrite the TSV. Audio and labels are kept aligned *by
> construction*, not by luck.

---

## 3. What this approach is anchored on (the papers, and their exact roles)

Three pieces of prior work do specific jobs. It helps to be precise about what each one
actually contributes, because the project leans on each differently.

### 3.1 Basic Pitch — *the model we train* (Bittner et al., Spotify, ICASSP 2022)
Defines the **target** and the **constraints**:
- It consumes audio at **16 kHz** and produces frame-level **onset**, **frame (note)**, and
  **contour** activations on a Constant-Q front-end anchored at **A0 = 27.5 Hz**.
- Its note decoder is valid for **C1–C8 (MIDI 24–108)**. Our source scores are MIDI 29–83,
  so everything we generate lands in range.
- Its pitch convention is the standard `f = 440·2**((midi−69)/12)`. We use the *identical*
  formula when building melodies, so there is no octave/reference mismatch.

### 3.2 The WORLD vocoder — *the engine that decouples pitch from timbre* (Morise et al., 2016)
WORLD decomposes any voice recording into **three independent layers**:
- **F0** — the pitch contour (estimated by DIO/Harvest),
- **spectral envelope (`sp`)** — the timbre / formants (CheapTrick),
- **aperiodicity (`ap`)** — breathiness/noise (D4C).

Crucially, you can **change one layer and resynthesize** without disturbing the others.
*Role:* this is what lets us keep a speaker's timbre while replacing their pitch — and it
*guarantees* that swapping F0 doesn't move the formants. Pitch/timbre independence is the
whole reason Method 2 can satisfy the CRITICAL CONSTRAINT.

### 3.3 V2S, "Voice-to-Singing" — *the technique we adapt* (arXiv:2102.08575, Basak et al., 2021)
This paper (full title: *End-to-End Lyrics Recognition with Voice to Singing Style
Transfer*) introduced using WORLD to turn **speech into singing** by **F0 replacement**, and
established a key, non-obvious result we rely on:

> **No time-alignment is needed.** They explicitly do *not* align the F0 track to the
> speech. The spoken phonemes simply ride underneath; *all* of the pitch and timing comes
> from the imposed F0.

*Role:* it validates Method 2's mechanism and removes the hardest engineering problem
(aligning lyrics/phonemes to notes — which we couldn't do anyway, since we have no lyrics).

**Our one adaptation.** In the paper, the F0 is borrowed from a *real singing recording*
(their task is lyrics ASR, where exact pitch labels don't matter). For us the F0 **is** the
ground truth, so we cannot borrow it — we **author the F0 contour from the TSV's MIDI
values** instead. Everything else follows the paper. (We deliberately did *not* use Saitou
2007's speech-to-singing model: it spends its complexity on F0 micro-dynamics — overshoot,
vibrato, preparation — that we don't need and that would only risk nudging pitch away from
the labels.)

---

## 4. The pipeline, end to end

```
 scores/*.tsv ──┐
 (onset/offset/ │   Phase A · Method 2 (offline, shardable)
   MIDI)        │   scripts/A/vocal_synth_method_2.py
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ pick N LibriTTS speakers (f0-matched to the score's pitch)   │
   │ for each speaker:                                            │
   │   WORLD decompose speech  → f0_speech, sp, ap                │
   │   build F0 from TSV MIDI   → f0_score (silence in gaps)      │
   │   WORLD resynthesize(sp, ap, f0_score) → dry "singing"       │
   │   mute inter-note gaps, peak-normalize, 16 kHz int16         │
   └─────────────────────────────────────────────────────────────┘
                ▼
 syntheticdataset/<scoreid>_m2_<speaker>/{audio.wav, score.tsv}
                │
                │   Phase C · band-limiting augmentation (on-the-fly, training only)
                │   src/augmentation.py  ←called by→  src/lightning_module.py:training_step
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ per file, band-limit to its own vocal range (p = 0.6)        │
   │   keeps onset/offset/pitch labels valid (no time/pitch shift)│
   └─────────────────────────────────────────────────────────────┘
                ▼
   Phase D · train Basic Pitch (experiments_sample.sh) → best COnPOff_f1 checkpoint
```

### Phase A — Method 2: Speech-to-Singing synthesis
**Script:** `scripts/A/vocal_synth_method_2.py`. For each `score.tsv`:

1. **Choose speakers.** From a pool of LibriTTS `dev-clean` speakers (auto-downloaded on
   first run), pick **3 by default**, using **f0-match**: rank speakers by how little their
   natural pitch must be stretched to reach the score's median pitch, using an
   *octave-folded* distance (a speaker an octave away is as good as one already in range —
   WORLD handles octave shifts cleanly; awkward fractional stretches are what create thin,
   weak harmonics). We then sample from the best-matched pool to keep timbre variety.
2. **Cover the score.** A LibriTTS clip is a few seconds; scores run several minutes, so we
   concatenate one speaker's utterances to cover the whole duration (consistent timbre).
3. **Decompose → re-pitch → resynthesize.**
   ```python
   f0_speech, sp, ap = pw.wav2world(speech, sr)   # speaker's timbre + breathiness
   f0_score          = tsv_to_f0(notes, sr)        # our melody: per-note Hz, 0 in gaps
   singing           = pw.synthesize(f0_score, sp, ap, sr)
   ```
   `f0_score` is built on WORLD's 5 ms frame grid: each note `[onset, offset)` is filled
   with `440·2**((midi−69)/12)`; frames between notes are 0 (unvoiced).
4. **Clean up & write.** Mute inter-note gaps (so silences match the offsets), peak-normalize,
   write **16 kHz mono int16** `audio.wav` and a `score.tsv` that begins with the mandatory
   `# onset,offset,note` header followed by the **unchanged** note rows.

**Why pitch is exact:** the only pitched layer in the output is `f0_score`, which we
generate straight from the TSV. The speech contributes formants (`sp`) and breathiness
(`ap`) only — both pitch-independent. There is no pitch flag that can desync. A built-in
`--self-check` re-estimates the output F0 and reports per-note error: in practice the
**median error is ~3 cents** (50 cents = a quarter of the matching tolerance; 100 cents =
one semitone).

**What the audio sounds like (and why that's fine).** It sounds like a person *speaking* a
melody, with clear silences between notes and a fairly high pitch. That's expected, not a
bug: (a) the "breaks" are the rests in the score — 23–56% of every track is genuine silence,
which the model needs in order to learn offsets; (b) the pitch is high because sung melodies
(MIDI 60–72 ≈ 262–523 Hz) sit well above a speaking voice, and we keep the speaker's low
formants while raising only the pitch. For *transcription* training, perceptual prettiness
is irrelevant — only the onset/offset/pitch alignment matters, and that is exact.

### Phase C — Band-limiting augmentation (training time)
**File:** `src/augmentation.py`; **wired in** at `src/lightning_module.py::training_step`
(`audio = augment_audio(batch["audio"], batch["frame"])`, before the forward pass).

This is a *lightweight, label-preserving* augmentation applied **on the fly during training**
(not pre-rendered to disk). Per file, it band-limits the audio to **that file's own vocal
range**, read from the frame labels:

- **top edge** = 2 octaves above the highest active note (`f_max·4`),
- **bottom edge** = randomized 1–2 octaves below the lowest active note (`f_min / 2**U(1,2)`),
- realized as a pinned `BandPassFilter` + `LowPassFilter` (audiomentations).

It fires on a fraction **p = 0.6** of training samples (the rest pass through full-band, so
the model still sees plenty of clean vocals); files with no notes are left untouched. Because
band-pass/low-pass filtering shifts neither **time** nor **pitch**, the onset/offset/pitch
labels stay valid — it respects the CRITICAL CONSTRAINT. It runs only in `training_step`, so
validation, test, and the hidden set always see raw audio.

> **Scope, honestly:** this is a *narrower* Phase C than the originally-planned offline
> reverb/EQ/compression + accompaniment mixing. It improves robustness to bandwidth
> variation but does not, on its own, close the gap to full commercial mixes — see
> *Limitations* below.

### Phase D — Training Basic Pitch
**Launcher:** `MML26-singing-synthesis/experiments_sample.sh` → `python3 -m src.train`.
- **Train** on `Synthetic` (our `syntheticdataset/`), **validate** on `Klangio`.
- 8-second chunks, batch 32, `frame-weight 9`, `onset-weight 18`, lr 1e-4, precision 32.
- **Eval metric `COnPOff_f1`**; the best such checkpoint is saved automatically.
- `--limit-train-batches 200` caps each epoch (~6,400 chunks) so epochs stay short while
  `shuffle=True` keeps rotating fresh chunks from the whole dataset across epochs.
- **Early stopping (patience 10)** ends training at the metric plateau, so `--max-epochs 100`
  is just a ceiling — you typically converge in far fewer.
- The augmentation runs automatically inside training; there is no separate flag.

---

## 5. Distributed synthesis (split the heavy lifting across machines)

Generating ~1,200 multi-minute vocals (400 scores × 3 speakers) is the expensive part
(several hours single-threaded; WORLD is CPU-only). It's embarrassingly parallel, so the
synthesizer supports **sharding**:

```bash
# 4 machines — run ONE per machine, changing only the shard index:
python3 scripts/A/vocal_synth_method_2.py --shard 0/4 --skip-existing
python3 scripts/A/vocal_synth_method_2.py --shard 1/4 --skip-existing
python3 scripts/A/vocal_synth_method_2.py --shard 2/4 --skip-existing
python3 scripts/A/vocal_synth_method_2.py --shard 3/4 --skip-existing
```

- `--shard I/K` strides the score list so each shard gets a balanced mix of lengths and the
  shards together cover everything with no overlap.
- **Speaker choice is deterministic per score**, so shards (and resumed runs) agree without
  coordination; output folders `<scoreid>_m2_<speaker>` never collide.
- `--skip-existing` resumes an interrupted run.
- **Merge** is just a copy: drop every machine's `syntheticdataset/` into one.

A GPU does **not** help synthesis (pure DSP); a GPU *does* help training. On Colab, run
generation on CPU notebooks and training on a GPU notebook.

---

## 6. How to run

### 6.0 Setup
```bash
cd MML26-singing-synthesis
pip install -r requirements.txt        # includes pyworld + audiomentations
```

### 6.1 Generate data (Phase A)
From the repo root (first run auto-downloads LibriTTS dev-clean, ~1.2 GB):
```bash
# whole dataset, defaults = 3 speakers/score, 16 kHz int16, f0-match
python3 scripts/A/vocal_synth_method_2.py

# quick smoke test (2 scores) with pitch self-check
python3 scripts/A/vocal_synth_method_2.py --limit 2 --self-check
```
Output lands in `MML26-singing-synthesis/syntheticdataset/<scoreid>_m2_<speaker>/`.
Useful flags: `--n-speakers N`, `--shard I/K`, `--start N`, `--limit N`, `--skip-existing`,
`--self-check`, `--no-f0-match`.

### 6.2 Train (Phase D)
```bash
cd MML26-singing-synthesis
./experiments_sample.sh            # add --accelerator mps|cpu locally; default is gpu
```
- Watch live in **wandb** (project `basic_pitch+_MML-hackaton2026`) when launched with
  `--logger wandb`; you must be logged in *on the machine that trains* (`wandb login` /
  `WANDB_API_KEY`).
- Or view TensorBoard logs: `tensorboard --logdir ./BASIC_PITCH_CHALLENGE/`.

### 6.3 Inference (listen / transcribe)
```bash
python -m src.inference --checkpoint-path <ckpt> --input-path /path/to/audio --output-dir predictions
```

---

## 7. Key implementation facts (and the gotchas they prevent)
- **Sample rate is 16 kHz** (`src/constants.py`), not 22050. The dataloader resamples
  anything, but we write 16 kHz int16 directly (smaller, no quality loss for this model).
- **`score.tsv` needs a header.** The loader uses `np.loadtxt(..., skiprows=1)`; a missing
  header drops the first note. We always write `# onset,offset,note` first.
- **Pitch must be MIDI 24–108** to be decodable; our scores (29–83) are safe.
- **MIDI→Hz matches the model's CQT** (A0 = 27.5 Hz), so no reference/octave mismatch.
- **Storage:** 22050/float32 ≈ 17 MB/file; **16 kHz/int16 ≈ 6 MB/file** (~8–9 GB for
  400×3). FLAC would be ~2.4 MB but the loader hard-codes `audio.wav`, so we keep WAV.

---

## 8. Design decisions & trade-offs
- **Method 2 over Method 1** — speech-to-singing gives many real-voice timbres "for free"
  from LibriTTS, and pitch is exact by construction with no synthesizer pitch flag to drift.
- **3 speakers/score + f0-match** — melodic diversity (400 distinct songs) matters more for
  transcription than timbre count; f0-match avoids the over-stretched, weak-harmonic voices.
- **16 kHz int16** — cuts storage ~3× vs the original 22050/float32 with no model-relevant
  quality loss (the model is 16 kHz anyway).
- **On-the-fly augmentation** (vs pre-rendering) — fresh random band-limit each epoch, no
  disk cost, training-only, uses the sanctioned `training_step` hook (no dataloader edits).
- **`--limit-train-batches`** — bounds per-epoch wall-clock while `shuffle=True` keeps all
  data in rotation; cleaner than physically deleting files.

---

## 9. Limitations & future work
- **No Method 1** (`midi2voice` / RVC singing synthesis) — only V2S/WORLD vocals.
- **No Phase B** (AI-generated accompaniment / instrumentals) and **no full offline Phase C**
  (reverb/EQ/compression + mixing lead vocals with instrumentals at varying SIR). Today we
  train on **dry, band-limited vocals**, so the "implicit source separation" idea
  (mixed audio paired with clean labels) is **not yet exercised** — this is the main gap to
  real commercial mixes and the highest-value next step.
- **Performance** — WORLD synthesis is CPU-only; the training-time augmentation does a
  per-sample CPU round-trip. Both are workable but not fast.

---

## 10. Repository layout
```
voc-aI/
├── CLAUDE.md                          # working project context (current state)
├── README.md                          # this file
├── scripts/A/vocal_synth_method_2.py  # Phase A · Method 2 synthesizer (usage at top of file)
└── MML26-singing-synthesis/           # the challenge repo (Basic Pitch + training)
    ├── scores/*.tsv                   # source note annotations (INPUT)
    ├── klangiodataset/                # validation/test singing data
    ├── syntheticdataset/              # our generated training data (OUTPUT)
    ├── src/
    │   ├── augmentation.py            # Phase C · band-limiting augmentation
    │   ├── lightning_module.py        # training_step calls augment_audio
    │   ├── train.py / constants.py / basic_pitch_model.py / dataloading/
    ├── experiments_sample.sh          # training launcher
    └── requirements.txt               # + pyworld, audiomentations
```

---

## 11. References
- Z. Bittner et al., *A Lightweight Instrument-Agnostic Model for Polyphonic Note
  Transcription and Multipitch Estimation* (Basic Pitch), ICASSP 2022.
- M. Morise, F. Yokomori, K. Ozawa, *WORLD: A Vocoder-Based High-Quality Speech Synthesis
  System for Real-Time Applications*, IEICE Trans. 2016.
- S. Basak, S. Agarwal, S. Ganapathy, N. Takahashi, *End-to-End Lyrics Recognition with
  Voice to Singing Style Transfer* (V2S), arXiv:2102.08575, 2021.
