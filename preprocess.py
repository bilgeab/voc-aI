"""
Dataset preprocessing / generation pipeline.

This orchestrates the synthetic-data pipeline described in CLAUDE.md and builds a
training dataset on disk in the layout the MML26 model expects:

    <output-dir>/
        config.json            <- records exactly which techniques produced this dataset
        <some_unique_id>/
            audio.wav
            score.tsv
        ...

The three phases:

    Phase A  Multi-speaker vocal synthesis (timbre augmentation).
             MUTUALLY EXCLUSIVE: pick exactly one option.
                a -> scripts/A/option_a.py   (e.g. midi2voice parameter randomization)
                b -> scripts/A/option_b.py   (e.g. RVC voice conversion)

    Phase B  Accompaniment & harmony generation.
             ADDITIVE: pick zero or more.
                instrumentals -> scripts/B/instrumentals.py
                harmonies     -> scripts/B/harmonies.py

    Phase C  Augmentation & mixing.
             ADDITIVE: pick zero or more.
                reverb -> scripts/C/reverb.py
                eq     -> scripts/C/eq.py
                mix    -> scripts/C/mix.py

Usage:
    python preprocess.py \
        --scores MML26-singing-synthesis/scores \
        --output-dir syntheticdataset-exp/dataset-rvc-pop \
        --phase-a b \
        --phase-b instrumentals harmonies \
        --phase-c reverb eq mix
"""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PHASE_A_OPTIONS = {
    "a": PROJECT_ROOT / "scripts" / "A" / "option_a.py",
    "b": PROJECT_ROOT / "scripts" / "A" / "option_b.py",
}

PHASE_B_OPTIONS = {
    "instrumentals": PROJECT_ROOT / "scripts" / "B" / "instrumentals.py",
    "harmonies": PROJECT_ROOT / "scripts" / "B" / "harmonies.py",
}

PHASE_C_OPTIONS = {
    "reverb": PROJECT_ROOT / "scripts" / "C" / "reverb.py",
    "eq": PROJECT_ROOT / "scripts" / "C" / "eq.py",
    "mix": PROJECT_ROOT / "scripts" / "C" / "mix.py",
}


def _load_module(path: Path):
    """Import a technique module by file path (the scripts/ dirs are not packages)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Technique script not found: {path}\n"
            f"This option is registered but its implementation file is missing. "
            f"Create it with a `run(ctx)` entry point."
        )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_technique(label: str, path: Path, ctx: dict) -> None:
    """Dispatch to a single technique module's `run(ctx)`."""
    print(f"\n[preprocess] >>> {label}: {path.relative_to(PROJECT_ROOT)}")
    module = _load_module(path)
    if not hasattr(module, "run"):
        raise AttributeError(
            f"{path} must define `run(ctx: dict) -> None`. "
            f"This is the agreed entry point for technique modules."
        )
    module.run(ctx)


def main():
    parser = argparse.ArgumentParser(
        description="Build a synthetic SVT training dataset from scores using selectable techniques.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scores",
        type=str,
        default="MML26-singing-synthesis/scores",
        help="Directory of source .tsv scores (onset/offset/pitch) to synthesize from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write the generated dataset (and its config.json)",
    )

    parser.add_argument(
        "--phase-a",
        type=str,
        required=True,
        choices=list(PHASE_A_OPTIONS.keys()),
        help="Vocal synthesis technique (mutually exclusive: choose one)",
    )

    parser.add_argument(
        "--phase-b",
        type=str,
        nargs="*",
        default=[],
        choices=list(PHASE_B_OPTIONS.keys()),
        help="Accompaniment/harmony techniques (additive: choose any number)",
    )

    parser.add_argument(
        "--phase-c",
        type=str,
        nargs="*",
        default=[],
        choices=list(PHASE_C_OPTIONS.keys()),
        help="Augmentation/mixing techniques (additive: choose any number, applied in order)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation",
    )

    args = parser.parse_args()

    scores_dir = Path(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scores_dir.is_dir():
        raise FileNotFoundError(f"--scores directory does not exist: {scores_dir}")

    # Shared context handed to every technique module. Each module reads what it
    # needs and writes into output_dir following the audio.wav / score.tsv layout.
    ctx = {
        "scores_dir": scores_dir,
        "output_dir": output_dir,
        "seed": args.seed,
        "phase_a": args.phase_a,
        "phase_b": list(args.phase_b),
        "phase_c": list(args.phase_c),
    }

    print("=" * 70)
    print("PREPROCESS / DATASET GENERATION")
    print("=" * 70)
    print(f"Scores:     {scores_dir}")
    print(f"Output:     {output_dir}")
    print(f"Phase A:    {args.phase_a}  (vocal synthesis)")
    print(f"Phase B:    {args.phase_b or '(none)'}  (accompaniment/harmony)")
    print(f"Phase C:    {args.phase_c or '(none)'}  (augmentation/mixing)")
    print(f"Seed:       {args.seed}")
    print("=" * 70)

    # --- Dispatch -------------------------------------------------------------
    # Phase A first (produces base vocals), then B (adds backing/harmonies),
    # then C (augments/mixes the result), each phase in the listed order.
    _run_technique(f"Phase A [{args.phase_a}]", PHASE_A_OPTIONS[args.phase_a], ctx)

    for choice in args.phase_b:
        _run_technique(f"Phase B [{choice}]", PHASE_B_OPTIONS[choice], ctx)

    for choice in args.phase_c:
        _run_technique(f"Phase C [{choice}]", PHASE_C_OPTIONS[choice], ctx)

    # --- Provenance: config.json saved next to the dataset --------------------
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scores": str(scores_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "phase_a": args.phase_a,
        "phase_b": list(args.phase_b),
        "phase_c": list(args.phase_c),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("PREPROCESS COMPLETE")
    print("=" * 70)
    print(f"Dataset:  {output_dir}")
    print(f"Config:   {config_path}")
    print("Next:     python train.py --dataset-path", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
