# Project Context: Singing Voice Transcription with Synthetic Data

## 1. Project Objective
We are building a highly robust Singing Voice Transcription (SVT) model capable of predicting the **onset**, **offset**, and **pitch** of lead vocals in commercial pop songs. 
Because manual annotation of commercial music is expensive, we are using a **Synthetic Data Generation and Mixing Pipeline**. We will train the model on noisy, artificially mixed audio but calculate the loss against clean, single-vocal ground truth (implicit source separation).

## 2. Reference Repository
We will be using the model architecture and training scripts provided in this repository:
[https://github.com/Klangio/MML26-singing-synthesis/blob/main/README.md](https://github.com/Klangio/MML26-singing-synthesis/blob/main/README.md)

The provided transcription model is a (slightly modified) **Basic Pitch** CNN. Two facts from the repo constrain everything downstream:
- **Sample rate:** Basic Pitch operates at **22050 Hz**. Every generated `.wav` (both Phase A methods, and the final mix) must be produced or resampled to 22050 Hz.
- **Data format / dataloader:** The provided dataloader already iterates `syntheticdataset/<unique_id>/` looking for `audio.wav` + `score.tsv` pairs. We feed it data by writing that structure — we do **not** need to (and per challenge rules should not) modify the training code.

## 3. The Data Pipeline (What needs to be implemented)

### Phase A: Multi-Speaker Vocal Synthesis (Timbre Augmentation ONLY)
We have source `.tsv` files containing purely structural data: `onset`, `offset`, and `pitch`. 
**Task:** Convert each `.tsv` into multiple distinct raw `.wav` files to achieve a massive dataset multiplier and teach the model timbre invariance. 
**CRITICAL CONSTRAINT:** We must NEVER alter the fundamental pitch of the generated audio unless we also rewrite the `.tsv`. Therefore, focus strictly on **Timbre Augmentation** (altering the tone/resonance while keeping pitch identical).
*Goal:* From one TSV, produce 5-10 distinct vocal tracks.

> **Phase A has TWO interchangeable synthesis methods.** Both produce **dry, pitch-exact vocals** in the same output format (`audio.wav` + `score.tsv` per folder), so they can be used independently or *together* to maximize timbre diversity. Method 1 is a score-driven singing synthesizer; Method 2 transfers a real speaker's voice onto the score's melody. Running both over the same TSVs is the cheapest way to broaden the timbre distribution the model sees.

#### Method 1 — Score-Driven Singing Synthesis (`midi2voice`)
Synthesize directly from the score using a singing voice synthesizer, varying voice parameters to produce distinct timbres at identical pitch/timing.

1.  **midi2voice Parameter Randomization:** Write a batch script to run `midi2voice` multiple times per TSV using randomized flags:
    * `-g`: Swap between `female` and `male`.
    * `-s` (Formant/Vocal Tract): Randomize between `-0.8` and `0.8` to change the physical resonance of the voice (does NOT change pitch).
    * `-v` (Vibrato): Randomize between `0` and `2` to vary the singing style.
    * *(DO NOT USE the `-p` pitch shift flag, as it will desync from the TSV).*
2.  **AI Voice Conversion (RVC):** Feed the base `midi2voice` outputs through Retrieval-based Voice Conversion (RVC) using community models to instantly generate highly realistic, distinct timbres while maintaining the exact ground truth pitch/timing.
3.  **DSP Modification:** Use `librosa` or `Parselmouth` to apply algorithmic **formant shifting** (altering throat size representation without altering pitch).

#### Method 2 — Speech-to-Singing Transfer (V2S / WORLD vocoder)
Borrow a real human voice's *texture* from a speech corpus and inject the score's *melody* into it. The WORLD vocoder decomposes any speech recording into three independent layers — **spectral envelope** (timbre), **F0 contour** (melody), and **aperiodicity** (breathiness). We keep a LibriTTS speaker's envelope and aperiodicity, **discard their speaking F0, and replace it with an F0 contour we build directly from the TSV.** Resynthesizing makes that speaker "sing" the score.

* **Pitch is exact by construction.** Because the F0 contour is generated from the TSV's MIDI values (`f0 = 440 * 2**((midi - 69) / 12)` over each note's onset–offset span), this method satisfies the CRITICAL CONSTRAINT *natively* — there is no separate pitch flag that could desync. The melody the model is graded on is the one layer we author by hand.
* **Diversity for free.** LibriTTS has ~2,300 speakers. Running each score through N speakers yields N distinct, pitch-identical "singers" — easily exceeding the 5-10 target with no extra modeling.
* **Pure timbre augmentation.** Different speakers = different timbres, same pitch/timing — exactly the Phase A goal.
* **Implementation:** `pyworld` (`pw.wav2world` to decompose, `pw.synthesize` to recombine) over a pool of LibriTTS utterances. Output at **22050 Hz** to match Basic Pitch.

```python
import pyworld as pw
f0_speech, sp, ap = pw.wav2world(speech, sr)   # decompose a LibriTTS speaker
f0_score = tsv_to_f0(tsv_notes, sr)            # build melody from the TSV (MIDI -> Hz)
singing = pw.synthesize(f0_score, sp, ap, sr)  # that speaker's voice, our melody
```

> **Note on key/tempo variety:** If we want the *same melody* in different keys or tempos, we build that into the F0 contour (shift MIDI values, scale note timings) **and rewrite the TSV to match**, before synthesis — keeping audio and labels aligned by construction. This stays within the CRITICAL CONSTRAINT.

### Phase B: Accompaniment & Harmony Generation
We must bridge the domain gap between "dry synthetic vocals" and "commercial pop mixes". 
**Task:** Generate diverse backing tracks and mix them with the vocals.
* **Instrumentals:** Use generative AI (e.g., MusicGen, Mustango, or an API) to generate royalty-free, full instrumental backing tracks (drums, bass, synths, guitars) across various genres.
* **Harmonies:** If adding harmonies, generate them using a *different* vocal synthesizer (or different RVC model) than the lead vocal so the model learns to separate distinct voices.

### Phase C: Data Augmentation & Mixing
**Task:** Apply realistic audio degradations and mix the stems.
* **Modulation:** Add randomized micro-delays (10-30ms) to the accompaniments/harmonies. 
* **Studio FX:** Apply random Room Impulse Responses (algorithmic reverb), multiband compression, and EQ curves to the vocals.
* **The Final Mix:** Mix the augmented lead vocal `.wav` with the AI-generated instrumental `.wav` at varying Signal-to-Interference Ratios (e.g., -5dB to +5dB).

### Phase D: Model Training
**Task:** Provide the mixed data to the existing training pipeline in the expected format.
* **Input:** The heavily augmented, fully mixed `.wav` (Lead Vocal + Instrumentals + FX), written as `audio.wav`.
* **Ground Truth:** The original, clean `.tsv` (Lead Vocal Onset, Offset, Pitch ONLY), written as `score.tsv`.
* *Mechanism:* By penalizing the model for predicting instrumental notes, it will naturally learn to perform implicit source separation and ignore background noise.
* *Note:* The provided dataloader already consumes `syntheticdataset/<id>/{audio.wav, score.tsv}`. Implicit source separation is achieved purely by pairing **mixed audio** with the **clean TSV** in that standard layout — no modification of the training/dataloader code is required (and the challenge rules forbid modifying the provided model code). Training is launched with `./experiments_sample.sh`.

## 4. Immediate Tasks for Claude
When reading this file, please prepare to help me write the following Python modules:
1.  `vocal_synth_method_1.py`: A script to batch-generate 5-10+ diverse **dry** vocal WAVs (at 22050 Hz) from each TSV:
    * **Method 1:** automate `midi2voice` (using `-g`, `-s`, `-v` ONLY — no pitch shifting) plus optional RVC/DSP timbre modification.
2. `vocal_synth_method_2.py`: A script to batch-generate 5-10+ diverse **dry** vocal WAVs (at 22050 Hz) from each TSV:
    * **Method 2:** V2S/WORLD speech-to-singing transfer, building the F0 from the TSV and resynthesizing over a pool of LibriTTS speakers (`pyworld`).
    The two methods share the same output contract (`audio.wav` + `score.tsv` per folder) so their outputs are interchangeable and can be pooled.
3. `accompaniment_gen.py`: A script to automate the generation of background instrumentals.
4. `mix_and_augment.py`: An audio processing pipeline using `librosa`, `audiomentations`, or `pydub` to apply reverb, EQ, and mix the vocal/instrumental stems at random SIR levels.
5. `dataset_loader.py`: Write the paired `mixed .wav` + original `.tsv` data into the `syntheticdataset/<id>/` layout the provided Klangio dataloader already expects (rather than altering the dataloader itself).