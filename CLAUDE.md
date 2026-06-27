# Project Context: Singing Voice Transcription with Synthetic Data

## 1. Project Objective
We are building a highly robust Singing Voice Transcription (SVT) model capable of predicting the **onset**, **offset**, and **pitch** of lead vocals in commercial pop songs. 
Because manual annotation of commercial music is expensive, we are using a **Synthetic Data Generation and Mixing Pipeline**. We will train the model on noisy, artificially mixed audio but calculate the loss against clean, single-vocal ground truth (implicit source separation).

## 2. Reference Repository
We will be using the model architecture and training scripts provided in this repository:
[https://github.com/Klangio/MML26-singing-synthesis/blob/main/README.md](https://github.com/Klangio/MML26-singing-synthesis/blob/main/README.md)

## 3. The Data Pipeline (What needs to be implemented)

### Phase A: Multi-Speaker Vocal Synthesis (Timbre Augmentation ONLY)
We have source `.tsv` files containing purely structural data: `onset`, `offset`, and `pitch`. 
**Task:** Convert each `.tsv` into multiple distinct raw `.wav` files to achieve a massive dataset multiplier and teach the model timbre invariance. 
**CRITICAL CONSTRAINT:** We must NEVER alter the fundamental pitch of the generated audio unless we also rewrite the `.tsv`. Therefore, focus strictly on **Timbre Augmentation** (altering the tone/resonance while keeping pitch identical).
*Goal:* From one TSV, produce 5-10 distinct vocal tracks.

**Implementation Strategies for Phase A:**
1.  **midi2voice Parameter Randomization:** Write a batch script to run `midi2voice` multiple times per TSV using randomized flags:
    * `-g`: Swap between `female` and `male`.
    * `-s` (Formant/Vocal Tract): Randomize between `-0.8` and `0.8` to change the physical resonance of the voice (does NOT change pitch).
    * `-v` (Vibrato): Randomize between `0` and `2` to vary the singing style.
    * *(DO NOT USE the `-p` pitch shift flag, as it will desync from the TSV).*
2.  **AI Voice Conversion (RVC):** Feed the base `midi2voice` outputs through Retrieval-based Voice Conversion (RVC) using community models to instantly generate highly realistic, distinct timbres while maintaining the exact ground truth pitch/timing.
3.  **DSP Modification:** Use `librosa` or `Parselmouth` to apply algorithmic **formant shifting** (altering throat size representation without altering pitch).

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
**Task:** Modify the training dataloader to consume the new mixed data.
* **Input:** The heavily augmented, fully mixed `.wav` (Lead Vocal + Instrumentals + FX).
* **Ground Truth:** The original, clean `.tsv` (Lead Vocal Onset, Offset, Pitch ONLY).
* *Mechanism:* By penalizing the model for predicting instrumental notes, it will naturally learn to perform implicit source separation and ignore background noise.

## 4. Immediate Tasks for Claude
When reading this file, please prepare to help me write the following Python modules:
1.  `vocal_synth.py`: A script to automate `midi2voice` (utilizing `-g`, `-s`, `-v` parameters ONLY—no pitch shifting) and integrate RVC/DSP modifications to batch-process the TSV/MIDI files into 5-10 diverse dry vocal WAVs each.
2.  `accompaniment_gen.py`: A script to automate the generation of background instrumentals.
3.  `mix_and_augment.py`: An audio processing pipeline using `librosa`, `audiomentations`, or `pydub` to apply reverb, EQ, and mix the vocal/instrumental stems at random SIR levels.
4.  `dataset_loader.py`: Adapting the Klangio GitHub repository's dataloader to pair the mixed `.wav` files with the original `.tsv` files for training.