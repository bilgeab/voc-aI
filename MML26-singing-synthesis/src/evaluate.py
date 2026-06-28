"""
Standalone evaluation for a trained checkpoint.

Loads a checkpoint and runs the Lightning test loop over a registered dataset
(default: the provided real Klangio test set), printing the transcription
metrics (COnP_f1, COnPOff_f1, pitch_mse, ...) computed by `get_metrics_dict`.

Run from the repo root so the dataset's default relative path resolves:

    python -m src.evaluate --checkpoint-path /path/to/best.ckpt --accelerator cpu
"""

import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .lightning_module import LightningModuleSingingVoice
from .dataloading import get_dataset_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", required=True,
                        help="Path to the checkpoint to evaluate")
    parser.add_argument("--dataset", default="Klangio",
                        help="Registered dataset name (provided real test set)")
    parser.add_argument("--groups", nargs="+", default=["test"],
                        help="Dataset groups to evaluate on")
    parser.add_argument("--accelerator", default="cpu",
                        choices=["cpu", "gpu", "mps"],
                        help="Hardware accelerator")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Data loading workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    # weights_only=False: the checkpoint stores numpy scalars (thresholds),
    # which the PyTorch 2.6+ safe unpickler rejects. Safe here — it is the
    # user's own trained checkpoint.
    model = LightningModuleSingingVoice.load_from_checkpoint(
        args.checkpoint_path,
        weights_only=False,
    )
    print(f"Thresholds - Onset: {model.onset_threshold:.3f}, "
          f"Frame: {model.frame_threshold:.3f}")

    registry = get_dataset_registry()
    if args.dataset not in registry:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. Available: {list(registry)}"
        )
    dataset_cls = registry[args.dataset]["class"]
    dataset = dataset_cls(
        groups=args.groups,
        sequence_length=None,
        seed=42,
        device="cpu",
    )
    print(f"{args.dataset} dataset ({args.groups}): {len(dataset)} songs")
    if len(dataset) == 0:
        raise RuntimeError(
            f"No (audio, tsv) pairs found for {args.dataset} / {args.groups}."
        )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    results = trainer.test(model, loader)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for metrics in results:
        for name, val in sorted(metrics.items()):
            print(f"{name:40s} {val:.4f}")


if __name__ == "__main__":
    main()
