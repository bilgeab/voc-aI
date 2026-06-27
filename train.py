"""
Top-level training entry point for the challenge.

This wraps the MML26 Basic Pitch training pipeline so you can point it at any
dataset directory produced by `preprocess.py` (or hand-assembled). It:

  1. Trains the model on `--dataset-path` (Synthetic layout: <id>/audio.wav + score.tsv).
  2. Validates / tests on the Klangio dataset that ships with the MML26 repo.
  3. Logs the challenge metrics (COnPOff_f1, COnP_f1, COn_f1) to Weights & Biases.
  4. Saves the best checkpoint + run_config.json + metrics.json into a run
     directory derived from the dataset name (so every set of weights records
     exactly which data and settings produced it).

Usage:
    python train.py --dataset-path syntheticdataset-exp/dataset-rvc-pop

    # No wandb account / offline smoke test:
    python train.py --dataset-path <path> --wandb-mode disabled

macOS note: torchcodec (a torchaudio dependency) needs an older FFmpeg's shared
libs than the Homebrew default. Install `ffmpeg@7` (`brew install ffmpeg@7`)
and run with:
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib python train.py ...
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MML26_ROOT = PROJECT_ROOT / "MML26-singing-synthesis"

sys.path.insert(0, str(MML26_ROOT))

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.transcription_utils.evaluation import METRICS_REGISTRY  # noqa: E402
from src.lightning_module import LightningModuleSingingVoice  # noqa: E402
from src.constants import SAMPLE_RATE, WANDB_PROJECT  # noqa: E402
from src.dataloading import get_dataset_registry  # noqa: E402
from src.dataloading.synthetic_dataset import SyntheticDataset  # noqa: E402

VAL_DATASET_NAME = "Klangio"
TEST_DATASET_NAME = "Klangio"


def _resolve_accelerator(requested: str) -> str:
    """Pick the best available hardware accelerator (cuda > mps > cpu) when 'auto'."""
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Train Basic Pitch on a chosen dataset (Klangio val/test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- The one required argument: which dataset to train on -----------------
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the training dataset (Synthetic layout: <id>/audio.wav + score.tsv)",
    )
    data_group.add_argument(
        "--val-dataset",
        type=str,
        default=VAL_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for validation",
    )
    data_group.add_argument(
        "--val-groups",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to use for validation",
    )
    data_group.add_argument(
        "--test-dataset",
        type=str,
        default=TEST_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for post-training testing",
    )
    data_group.add_argument(
        "--test-groups",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to use for post-training testing",
    )
    data_group.add_argument(
        "--eval-metric",
        type=str,
        default="COnPOff_f1",
        choices=list(METRICS_REGISTRY.keys()),
        help="Metric to track to select the best checkpoint",
    )

    audio_group = parser.add_argument_group("Audio Processing")
    audio_group.add_argument("--sequence-length", type=float, default=8.0,
                             help="Length of audio chunks in seconds (None for full audio)")
    audio_group.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    audio_group.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 3) // 3),
                             help="Number of data loading workers")

    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    train_group.add_argument("--accumulate-gradients", type=int, default=1,
                             help="Number of gradients to accumulate")
    train_group.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                             help="Optimizer to use")
    train_group.add_argument("--onset-weight", type=float, default=18.0,
                             help="Weight for onset loss (positive class weight)")
    train_group.add_argument("--frame-weight", type=float, default=9.0,
                             help="Weight for frame loss (positive class weight)")
    train_group.add_argument("--max-epochs", type=int, default=100, help="Maximum number of training epochs")
    train_group.add_argument("--patience", type=int, default=10,
                             help="Early stopping patience (epochs without improvement)")
    train_group.add_argument("--limit-train-batches", type=int, default=None,
                             help="Limit training batches per epoch")
    train_group.add_argument("--limit-val-batches", type=int, default=None,
                             help="Limit validation batches per epoch")
    train_group.add_argument("--num-sanity-val-steps", type=int, default=2,
                             help="Number of sanity check validation steps (0 to disable)")

    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "mps"],
                              help="Hardware accelerator to use ('auto' picks cuda > mps > cpu)")
    system_group.add_argument("--precision", type=str, default="32", choices=["16", "bf16", "32"],
                              help="Training precision")
    system_group.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    system_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    system_group.add_argument("--runs-dir", type=str, default="runs",
                              help="Root directory under which per-dataset run dirs are created")
    system_group.add_argument("--experiment-name", type=str, default=None,
                              help="Run name (defaults to the dataset directory name)")
    system_group.add_argument("--wandb-mode", type=str, default="online",
                              choices=["online", "offline", "disabled"],
                              help="Weights & Biases mode ('disabled' for a no-account smoke test)")
    system_group.add_argument("--resume-from", type=str, default=None,
                              help="Path to a checkpoint to resume training from")

    args = parser.parse_args()

    args.accelerator = _resolve_accelerator(args.accelerator)
    print(f"Using accelerator: {args.accelerator}")

    pl.seed_everything(args.seed, workers=True)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"--dataset-path does not exist: {dataset_path}")

    # Run name defaults to the dataset folder name so weights map to their data.
    experiment_name = args.experiment_name or dataset_path.name

    # Read the dataset's provenance (written by preprocess.py) if present.
    data_config = None
    data_config_path = dataset_path / "config.json"
    if data_config_path.exists():
        with open(data_config_path) as f:
            data_config = json.load(f)

    sequence_length_samples = (
        None if args.sequence_length is None else int(args.sequence_length * SAMPLE_RATE)
    )

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:        BasicPitchTranscriber")
    print(f"Dataset:      {dataset_path}")
    if data_config:
        print(f"Data config:  A={data_config.get('phase_a')} "
              f"B={data_config.get('phase_b')} C={data_config.get('phase_c')}")
    print(f"Val/Test:     {args.val_dataset} / {args.test_dataset}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Seq length:   {args.sequence_length}s")
    print(f"LR:           {args.learning_rate}")
    print(f"Max epochs:   {args.max_epochs}")
    print(f"Eval metric:  {args.eval_metric}")
    print("=" * 70)

    # --- Datasets -------------------------------------------------------------
    print("\nLoading training dataset...")
    train_dataset = SyntheticDataset(
        path=str(dataset_path),
        groups=["train"],
        sequence_length=sequence_length_samples,
        seed=args.seed,
        device="cpu",
    )
    print(f"Training dataset: {len(train_dataset)} samples")
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No (audio.wav, score.tsv) pairs found under {dataset_path}. "
            f"Did you run preprocess.py to populate it?"
        )

    registry = get_dataset_registry()

    # Registry dataset classes default to a path relative to the MML26 repo
    # (e.g. "klangiodataset"); resolve it there explicitly so it works regardless
    # of the caller's cwd.
    def _registry_dataset(name, groups):
        cls = registry[name]["class"]
        default_relative_path = cls.__init__.__defaults__[0]
        return cls(
            path=str(MML26_ROOT / default_relative_path), groups=groups,
            sequence_length=None, seed=args.seed, device="cpu",
        )

    print("\nLoading validation dataset...")
    val_dataset = _registry_dataset(args.val_dataset, args.val_groups)
    print(f"Validation dataset: {len(val_dataset)} samples")

    print(f"\nLoading test dataset ({args.test_dataset})...")
    test_dataset = _registry_dataset(args.test_dataset, args.test_groups)
    print(f"Test dataset: {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_num_workers = min(args.num_workers, 4)
    test_loader_kwargs = {"batch_size": 1, "shuffle": False, "num_workers": test_num_workers}
    if test_num_workers > 0:
        test_loader_kwargs["multiprocessing_context"] = "fork"
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    print(f"\nTrain batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")

    # --- Model ----------------------------------------------------------------
    print("\nInitializing Basic Pitch model (random init)...")
    model = LightningModuleSingingVoice(
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        onset_weight=args.onset_weight,
        frame_weight=args.frame_weight,
    )

    # --- Output layout --------------------------------------------------------
    # runs/<experiment_name>/  ->  checkpoints/best.ckpt + run_config.json + metrics.json
    # wandb's own local cache (if any) lives outside this directory; only the
    # artifacts we actually compare runs by are kept here.
    run_dir = Path(args.runs_dir) / experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        name=experiment_name,
        save_dir=str(run_dir),
        mode=args.wandb_mode,
        log_model=False,
    )
    try:
        cfg = {k: v for k, v in vars(args).items()
               if isinstance(v, (int, float, str, bool, type(None)))}
        if data_config:
            cfg.update({f"data_{k}": v for k, v in data_config.items()
                        if isinstance(v, (int, float, str, bool, type(None)))})
        logger.experiment.config.update(cfg, allow_val_change=True)
    except Exception as e:
        print(f"Warning: could not update wandb config: {e}")

    monitor_metric = f"eval/{args.eval_metric}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor=monitor_metric, mode="max", patience=args.patience, verbose=True,
    )

    # Persist the full run config (training args + which data was used) next to the weights.
    run_config = {
        "experiment_name": experiment_name,
        "dataset_path": str(dataset_path),
        "data_config": data_config,
        "args": vars(args),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    precision_map = {"16": "16-mixed", "bf16": "bf16-mixed", "32": "32-true"}
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": precision_map[args.precision],
        "logger": logger,
        "callbacks": [checkpoint_callback, lr_monitor, early_stopping],
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 10,
        "deterministic": False,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "accumulate_grad_batches": args.accumulate_gradients,
        "enable_model_summary": True,
        "check_val_every_n_epoch": 1,
    }
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print(f"Checkpoints -> {checkpoint_callback.dirpath}")
    print(f"Run config  -> {run_dir / 'run_config.json'}")
    print("=" * 70 + "\n")

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)

    # --- Test on best checkpoint ---------------------------------------------
    best_model_path = checkpoint_callback.best_model_path
    print("\n" + "=" * 70)
    print(f"LOADING BEST CHECKPOINT FOR TESTING: {best_model_path}")
    print("=" * 70)
    if best_model_path:
        # weights_only=False: this checkpoint was just produced by this same run, so it's trusted.
        model = LightningModuleSingingVoice.load_from_checkpoint(best_model_path, weights_only=False)
    trainer.test(model, test_loader)

    # --- Persist final metrics next to the weights ----------------------------
    metrics = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in trainer.callback_metrics.items()
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.log_hyperparams(vars(args))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {best_model_path}")
    if checkpoint_callback.best_model_score is not None:
        print(f"Best {args.eval_metric}: {checkpoint_callback.best_model_score:.4f}")
    print(f"Metrics:         {run_dir / 'metrics.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
