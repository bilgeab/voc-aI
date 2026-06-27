#!/usr/bin/env python3
"""Phase A / Method 2 -- Speech-to-Singing (V2S) via the WORLD vocoder.

Borrows a real LibriTTS speaker's *timbre* (spectral envelope + aperiodicity) and
imposes an F0 contour authored directly from each score's TSV, producing dry,
pitch-exact "singing". Different speakers -> different timbres at identical
pitch/timing, which is exactly the Phase A goal (timbre augmentation only).

Method adapts arXiv:2102.08575 (Basak et al., "End-to-end lyrics Recognition with
Voice to Singing Style Transfer"): decompose speech with WORLD, keep the spectral
envelope + aperiodicity, drop the speaker's F0, and synthesize with a new F0. The
paper borrows F0 from a real singing recording; we instead BUILD the F0 from the
TSV MIDI, so pitch is exact by construction and matches the ground-truth labels.

Output contract (per folder):  syntheticdataset/<scoreid>_m2_<speaker>/
    audio.wav    dry vocal, peak-normalized
    score.tsv    "# onset,offset,note" header + the unchanged note rows

The score.tsv header is mandatory: the provided dataloader reads labels with
np.loadtxt(..., skiprows=1), so a missing header would silently drop the first note.

-----------------------------------------------------------------------------
QUICK START
-----------------------------------------------------------------------------
Run from the repo root. First run auto-downloads LibriTTS dev-clean (~1.2 GB).
Defaults: 3 speakers/score, 16 kHz int16 WAV, f0-matched speaker selection.

  # Generate the full dataset on one machine (~7-11 h single-threaded):
  python3 scripts/A/vocal_synth_method_2.py

DISTRIBUTED SYNTHESIS -- split the 400 scores across K devices with --shard I/K.
Shards are non-overlapping and cover everything; merge the output folders after.
Speaker choice is deterministic per score, so shards/resumes stay consistent.

  # 4 devices -- run ONE of these per machine (change only the shard index):
  python3 scripts/A/vocal_synth_method_2.py --shard 0/4 --skip-existing   # device 1
  python3 scripts/A/vocal_synth_method_2.py --shard 1/4 --skip-existing   # device 2
  python3 scripts/A/vocal_synth_method_2.py --shard 2/4 --skip-existing   # device 3
  python3 scripts/A/vocal_synth_method_2.py --shard 3/4 --skip-existing   # device 4

  # Then merge each device's MML26-singing-synthesis/syntheticdataset/ into one
  # (folder names never collide across shards):
  rsync -a <device>:.../MML26-singing-synthesis/syntheticdataset/ ./.../syntheticdataset/

  --skip-existing   resume an interrupted run (skips folders already written)
  --limit N         process only N scores (after shard/start) -- handy for a smoke test
  --n-speakers N    distinct singers per score (default 3)
  --self-check      report per-note pitch error in cents
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --- constants -------------------------------------------------------------
FRAME_PERIOD_MS = 5.0          # WORLD default analysis/synthesis frame period
A4_HZ = 440.0
DEFAULT_SR = 16000             # matches the model's SAMPLE_RATE; written as int16 PCM
PEAK_TARGET = 0.891            # ~ -1 dBFS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MML = _REPO_ROOT / "MML26-singing-synthesis"
DEFAULT_SCORES = _MML / "scores"
DEFAULT_OUT = _MML / "syntheticdataset"
DEFAULT_LIBRITTS = _REPO_ROOT / ".cache" / "libritts"


# --- score / F0 ------------------------------------------------------------
def midi_to_hz(midi):
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def _octave_distance(speaker_hz, target_hz):
    """Pitch stretch in octaves, folded to the nearest octave (0 = perfect, 0.5 = worst)."""
    if not speaker_hz or speaker_hz <= 0 or target_hz <= 0:
        return 99.0
    r = np.log2(target_hz / speaker_hz)
    return abs(r - round(r))


def load_notes(tsv_path):
    """Return (N, 3) array of [onset_s, offset_s, midi]. Source TSVs have no header."""
    arr = np.loadtxt(tsv_path, delimiter="\t")
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float64)


def build_f0_contour(notes, n_frames, frame_period_ms=FRAME_PERIOD_MS, style="step"):
    """Author the target F0 contour from the TSV.

    Swappable by design: ``style="step"`` lays a constant per-note pitch over each
    note's [onset, offset] span and leaves inter-note frames unvoiced (f0=0). Future
    styles (vibrato / glide / overshoot) plug in here without touching the pipeline.

    Returns (f0, voiced_mask), both length ``n_frames``.
    """
    if style != "step":
        raise NotImplementedError(f"F0 style {style!r} not implemented yet")
    fp_s = frame_period_ms / 1000.0
    f0 = np.zeros(n_frames, dtype=np.float64)
    for onset, offset, midi in notes:
        lo = max(0, min(int(round(onset / fp_s)), n_frames))
        hi = max(0, min(int(round(offset / fp_s)), n_frames))
        if hi > lo:
            f0[lo:hi] = float(midi_to_hz(midi))  # later note wins on overlap (mono lead)
    return f0, f0 > 0


# --- speaker pool ----------------------------------------------------------
class LibriTTSPool:
    """Indexes LibriTTS dev-clean utterances by speaker, downloading on first use."""

    def __init__(self, root=DEFAULT_LIBRITTS, sr=DEFAULT_SR, seed=0):
        self.root = Path(root)
        self.sr = sr
        self._rng = random.Random(seed)
        self._ensure_downloaded()
        self.by_speaker = self._index()
        self.speakers = sorted(self.by_speaker)
        self.gender = self._load_gender()
        self._mean_f0_cache = {}

    @property
    def _base(self):
        return self.root / "LibriTTS" / "dev-clean"

    def _ensure_downloaded(self):
        if self._base.is_dir() and any(self._base.glob("*/*/*.wav")):
            return
        print(f"[libritts] downloading dev-clean to {self.root} (~1.2 GB, first run only)...")
        import torchaudio  # imported lazily so the rest of the script has no hard dep

        self.root.mkdir(parents=True, exist_ok=True)
        torchaudio.datasets.LIBRITTS(str(self.root), url="dev-clean", download=True)

    def _index(self):
        by = {}
        for wav in self._base.glob("*/*/*.wav"):
            by.setdefault(wav.parent.parent.name, []).append(wav)
        if not by:
            raise RuntimeError(f"No LibriTTS wavs found under {self._base}")
        return by

    def _load_gender(self):
        f = self.root / "LibriTTS" / "SPEAKERS.txt"
        gender = {}
        if f.exists():
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and parts[0]:
                    gender[parts[0]] = parts[1]
        return gender

    def _load_wav(self, path):
        import librosa

        y, _ = librosa.load(str(path), sr=self.sr, mono=True)
        return y.astype(np.float64)

    def mean_f0(self, speaker_id):
        """Median voiced speaking F0 for a speaker, cached. Scans a few utterances
        until enough voiced frames are found (some clips are short/near-silent)."""
        if speaker_id not in self._mean_f0_cache:
            import pyworld as pw

            voiced_all = []
            for path in self.by_speaker[speaker_id][:3]:
                y = self._load_wav(path)
                if y.size:
                    f0, _ = pw.harvest(np.ascontiguousarray(y), self.sr)
                    voiced_all.append(f0[f0 > 0])
                    if sum(v.size for v in voiced_all) >= 200:
                        break
            voiced = np.concatenate(voiced_all) if voiced_all else np.array([])
            self._mean_f0_cache[speaker_id] = float(np.median(voiced)) if voiced.size else 0.0
        return self._mean_f0_cache[speaker_id]

    def select_speakers(self, n, target_hz=None, rng=None, gender_match=False, f0_match=True):
        """Choose n speakers for a score.

        ``rng`` should be a per-score random.Random so the choice is deterministic
        *for that score regardless of processing order* -- this lets independent
        shards/devices (and resumed runs) produce identical speaker assignments.

        With f0_match (default), rank speakers by how little their natural pitch must
        be stretched to reach the score's median pitch -- using an *octave-folded*
        distance, so a speaker an octave away ranks as cleanly as one already in range
        (WORLD handles octave shifts well; awkward fractional stretches are what hurt).
        We then sample n from the best-matched pool, preserving timbre variety while
        avoiding the over-stretched, weak-harmonic voices.
        """
        rng = rng or self._rng
        if f0_match and target_hz:
            ranked = sorted(self.speakers, key=lambda s: _octave_distance(self.mean_f0(s), target_hz))
            pool = ranked[: max(n * 3, n + 5)]
            rng.shuffle(pool)
            return pool[:n]
        if gender_match and self.gender:
            # No inherent target gender; ensure a balanced mix of M/F when metadata exists.
            males = [s for s in self.speakers if self.gender.get(s) == "M"]
            females = [s for s in self.speakers if self.gender.get(s) == "F"]
            rng.shuffle(males)
            rng.shuffle(females)
            mixed = [s for pair in zip(females, males) for s in pair]
            cands = mixed or list(self.speakers)
        else:
            cands = list(self.speakers)
            rng.shuffle(cands)
        return cands[:n]

    def speaker_audio(self, speaker_id, target_s):
        """Concatenate one speaker's utterances (looping if needed) to >= target_s."""
        paths = list(self.by_speaker[speaker_id])
        self._rng.shuffle(paths)
        target_n = int(np.ceil(target_s * self.sr)) + self.sr
        chunks, total, i = [], 0, 0
        while total < target_n and i < len(paths) * 50:
            y = self._load_wav(paths[i % len(paths)])
            if y.size:
                chunks.append(y)
                total += y.size
            i += 1
        if not chunks:
            raise RuntimeError(f"Speaker {speaker_id} yielded no audio")
        return np.concatenate(chunks)


# --- synthesis -------------------------------------------------------------
def _voiced_envelope(voiced, n_samples, sr, frame_period_ms, fade_ms=5.0):
    """Per-sample gain (1 voiced / 0 silent) from a per-frame voiced mask, fades smoothed."""
    fp_s = frame_period_ms / 1000.0
    idx = np.minimum((np.arange(n_samples) / (fp_s * sr)).astype(int), len(voiced) - 1)
    env = voiced[idx].astype(np.float64)
    w = max(1, int(fade_ms / 1000.0 * sr))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return np.clip(env, 0.0, 1.0)


def synthesize(notes, speech, sr, frame_period_ms=FRAME_PERIOD_MS, gap_mute=True):
    """Impose the TSV-derived F0 onto a speaker's spectral envelope/aperiodicity."""
    import pyworld as pw

    duration_s = float(notes[:, 1].max())
    n = int(round(duration_s * sr))
    speech = speech[:n] if speech.size >= n else np.pad(speech, (0, n - speech.size))
    speech = np.ascontiguousarray(speech, dtype=np.float64)

    f0_sp, sp, ap = pw.wav2world(speech, sr, frame_period=frame_period_ms)
    f0_score, voiced = build_f0_contour(notes, sp.shape[0], frame_period_ms)
    y = pw.synthesize(f0_score, sp, ap, sr, frame_period=frame_period_ms)

    if gap_mute:
        y = y * _voiced_envelope(voiced, len(y), sr, frame_period_ms)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak * PEAK_TARGET
    return y.astype(np.float32)


def self_check(y, sr, notes, frame_period_ms=FRAME_PERIOD_MS):
    """Re-estimate F0 of the synthesized audio; return per-note error in cents.

    Error is octave-folded (|cents| mod 1200, nearest octave) because the F0
    re-estimator (harvest) occasionally locks onto a harmonic/subharmonic on short
    notes -- those are estimator artifacts, not synthesis errors (the imposed F0 is
    exact by construction). Returns list of (midi, folded_cents_or_None).
    """
    import pyworld as pw

    f0, _ = pw.harvest(np.ascontiguousarray(y.astype(np.float64)), sr, frame_period=frame_period_ms)
    fp_s = frame_period_ms / 1000.0
    rows = []
    for onset, offset, midi in notes:
        lo, hi = int(round(onset / fp_s)), int(round(offset / fp_s))
        seg = f0[lo:hi]
        seg = seg[seg > 0]
        if seg.size == 0:
            rows.append((float(midi), None))
            continue
        cents = 1200.0 * np.log2(np.median(seg) / float(midi_to_hz(midi)))
        rows.append((float(midi), abs(cents - 1200.0 * round(cents / 1200.0))))
    return rows


# --- writer / batch --------------------------------------------------------
def write_pair(out_dir, score_id, speaker, y, sr, notes):
    d = Path(out_dir) / f"{score_id}_m2_{speaker}"
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "audio.wav"), y, sr, subtype="PCM_16")
    with open(d / "score.tsv", "w") as f:
        f.write("# onset,offset,note\n")
        for onset, offset, midi in notes:
            f.write(f"{onset:.6f}\t{offset:.6f}\t{midi:.6f}\n")
    return d


def run(args):
    scores = sorted(Path(args.scores_dir).glob("*.tsv"))
    if args.shard:
        idx, cnt = (int(x) for x in args.shard.split("/"))
        scores = scores[idx::cnt]  # strided => each shard gets a balanced mix of lengths
        print(f"[shard] {idx}/{cnt} -> {len(scores)} scores")
    if args.start:
        scores = scores[args.start:]
    if args.limit:
        scores = scores[: args.limit]
    if not scores:
        print(f"No .tsv scores found in {args.scores_dir}", file=sys.stderr)
        return 1

    pool = LibriTTSPool(root=args.libritts_root, sr=args.sr, seed=args.seed)
    print(f"[libritts] {len(pool.speakers)} speakers indexed")

    from tqdm import tqdm

    for score_path in tqdm(scores, desc="scores"):
        score_id = score_path.stem
        notes = load_notes(score_path)
        if notes.size == 0:
            continue
        target_hz = float(np.median(midi_to_hz(notes[:, 2])))
        rng = random.Random(f"{args.seed}:{score_id}")  # deterministic per score (shard/resume safe)
        speakers = pool.select_speakers(
            args.n_speakers, target_hz=target_hz, rng=rng,
            gender_match=args.gender_match, f0_match=args.f0_match,
        )
        for speaker in speakers:
            out = Path(args.out_dir) / f"{score_id}_m2_{speaker}"
            if args.skip_existing and (out / "audio.wav").exists() and (out / "score.tsv").exists():
                continue
            speech = pool.speaker_audio(speaker, float(notes[:, 1].max()))
            y = synthesize(notes, speech, args.sr)
            write_pair(args.out_dir, score_id, speaker, y, args.sr, notes)
            if args.self_check:
                rows = self_check(y, args.sr, notes)
                a = np.array([c for _, c in rows if c is not None])
                miss = sum(1 for _, c in rows if c is None)
                if a.size:
                    print(f"  {score_id}_m2_{speaker}: median |err|={np.median(a):.1f} cents, "
                          f"p90={np.percentile(a, 90):.1f}, within50c={(a < 50).mean() * 100:.0f}%, "
                          f"unvoiced notes={miss}/{len(rows)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--libritts-root", type=Path, default=DEFAULT_LIBRITTS)
    p.add_argument("--n-speakers", type=int, default=3, help="distinct singers per score")
    p.add_argument("--sr", type=int, default=DEFAULT_SR)
    p.add_argument("--limit", type=int, default=0, help="process only N scores (after shard/start)")
    p.add_argument("--start", type=int, default=0, help="skip the first N scores (after sharding)")
    p.add_argument("--shard", type=str, default=None, metavar="I/K",
                   help="process only shard I of K, strided: split work across devices, e.g. 0/4")
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip score/speaker folders already written")
    p.add_argument("--gender-match", action="store_true", help="balance M/F speakers (needs SPEAKERS.txt)")
    p.add_argument("--f0-match", action=argparse.BooleanOptionalAction, default=True,
                   help="prefer speakers whose natural pitch is closest (octave-folded) to the score")
    p.add_argument("--self-check", action="store_true", help="report per-note pitch error in cents")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
