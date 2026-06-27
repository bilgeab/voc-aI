"""
Training script for Basic Pitch.

Default layout: train on Synthetic, validate and test on Klangio.

Usage:
    python -m src.train --batch-size 8 --max-epochs 50

Run `python -m src.train --help` for all options.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from socket import gethostname

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from .transcription_utils.evaluation import METRICS_REGISTRY
from .lightning_module import LightningModuleSingingVoice
from .constants import SAMPLE_RATE, WANDB_PROJECT
from .dataloading import get_dataset_registry

TRAIN_DATASET_NAME = "Synthetic"
VAL_DATASET_NAME = "Klangio"
TEST_DATASET_NAME = "Klangio"


def create_datasets(
    groups: List[str],
    sequence_length: Optional[int],
    seed: int,
    device: str,
    dataset_name: str = TRAIN_DATASET_NAME,
):
    registry = get_dataset_registry()
    dataset_meta = registry[dataset_name]
    print(f"\nLoading {dataset_name} (default path)")
    print(f"  Groups: {groups}")
    return dataset_meta["class"](
        groups=groups,
        sequence_length=sequence_length,
        seed=seed,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Basic Pitch (Synthetic train, Klangio val/test by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--train-dataset",
        type=str,
        default=TRAIN_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for training",
    )
    data_group.add_argument(
        "--val-dataset",
        type=str,
        default=VAL_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for validation",
    )
    data_group.add_argument(
        "--train-groups",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to use for training",
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
        default="COnP_f1",
        choices=list(METRICS_REGISTRY.keys()),
        help="Metric to track to select the best checkpoint",
    )

    audio_group = parser.add_argument_group("Audio Processing")
    audio_group.add_argument(
        "--sequence-length",
        type=float,
        default=8.0,
        help="Length of audio chunks in seconds (None for full audio)",
    )
    audio_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    audio_group.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() // 3,
        help="Number of data loading workers",
    )

    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    train_group.add_argument(
        "--accumulate-gradients",
        type=int,
        default=1,
        help="Number of gradients to accumulate",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use",
    )
    train_group.add_argument(
        "--onset-weight",
        type=float,
        default=1.0,
        help="Weight for onset loss (positive class weight)",
    )
    train_group.add_argument(
        "--frame-weight",
        type=float,
        default=1.0,
        help="Weight for frame loss (positive class weight)",
    )
    train_group.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    train_group.add_argument(
        "--val-check-interval-hours",
        type=float,
        default=None,
        help="Validate every X hours of audio. If None, validates every epoch.",
    )
    train_group.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch",
    )
    train_group.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch",
    )
    train_group.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=2,
        help="Number of sanity check validation steps (0 to disable)",
    )

    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "mps"],
        help="Hardware accelerator to use",
    )
    system_group.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "bf16", "32"],
        help="Training precision",
    )
    system_group.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    system_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    system_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save logs",
    )
    system_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to output-dir)",
    )
    system_group.add_argument(
        "--experiment-name",
        type=str,
        default="basic_pitch_synthetic_data",
        help="Name for this experiment",
    )
    system_group.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Logger backend to use",
    )
    system_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    system_group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="Weights & Biases tags",
    )
    system_group.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    system_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )

    args = parser.parse_args()

    if args.sequence_length is None:
        sequence_length_samples = None
    else:
        sequence_length_samples = int(args.sequence_length * SAMPLE_RATE)

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: BasicPitchTranscriber")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset: {args.val_dataset}")
    print(f"Test dataset: {args.test_dataset}")
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
    print(f"Train groups: {args.train_groups}")
    print(f"Val groups: {args.val_groups}")
    print(f"Test groups: {args.test_groups}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length (s): {args.sequence_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 70)
    print()

    print("Loading training dataset...")
    train_dataset = create_datasets(
        groups=args.train_groups,
        sequence_length=sequence_length_samples,
        seed=args.seed,
        device="cpu",
        dataset_name=args.train_dataset,
    )
    print(f"Training dataset: {len(train_dataset)} samples")

    print("\nLoading validation dataset...")
    val_dataset = create_datasets(
        groups=args.val_groups,
        sequence_length=None,
        seed=args.seed,
        device="cpu",
        dataset_name=args.val_dataset,
    )
    print(f"Validation dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print(f"\nLoading test dataset ({args.test_dataset})...")
    test_dataset = create_datasets(
        groups=args.test_groups,
        sequence_length=None,
        seed=args.seed,
        device="cpu",
        dataset_name=args.test_dataset,
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(os.cpu_count() // 3, 4),
        multiprocessing_context='fork',
    )
    print(f"Test batches: {len(test_loader)}")

    print("\nInitializing Basic Pitch model...")
    model = LightningModuleSingingVoice(
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        onset_weight=args.onset_weight,
        frame_weight=args.frame_weight,
    )

    base_dir = args.checkpoint_dir if args.checkpoint_dir else args.output_dir

    if args.logger == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as exc:
            raise ImportError(
                "WandB logger requested but not available. Install wandb with: pip install wandb"
            ) from exc

        wandb_tags = [] if args.wandb_tags is None else list(args.wandb_tags)
        logger = WandbLogger(
            project=WANDB_PROJECT,
            name=args.wandb_name or args.experiment_name,
            tags=wandb_tags + [gethostname()],
            save_dir=base_dir,
            mode=args.wandb_mode,
            log_model=False,
        )
        try:
            config_dict = {
                k: v for k, v in vars(args).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            logger.experiment.config.update(config_dict, allow_val_change=True)
        except Exception as e:
            print(f"Warning: Could not update WandB config: {e}")
    else:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logger requested but not available. Install tensorboard."
            ) from exc
        logger = TensorBoardLogger(
            save_dir=base_dir,
            name=args.experiment_name,
            default_hp_metric=False,
        )

    output_dir = Path(base_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.logger == "wandb":
        checkpoint_dir = output_dir / logger.version / "checkpoints"
    elif hasattr(logger, "log_dir") and logger.log_dir is not None:
        checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    else:
        checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = f"eval/{args.eval_metric}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{global_step:08d}_{eval_COnP_f1:.4f}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        patience=args.patience,
        verbose=True,
    )

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
    }

    if args.val_check_interval_hours is not None:
        samples_per_hour = 3600 * SAMPLE_RATE
        samples_per_batch = args.batch_size * sequence_length_samples
        batches_per_hour = samples_per_hour / samples_per_batch
        val_check_batches = int(batches_per_hour * args.val_check_interval_hours)
        trainer_kwargs["val_check_interval"] = max(1, val_check_batches)
    else:
        trainer_kwargs["check_val_every_n_epoch"] = 1

    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Checkpoints will be saved to: {checkpoint_callback.dirpath}")
    print("=" * 70)
    print()

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from, weights_only=True)

    print("\n" + "=" * 70)
    print("LOADING BEST CHECKPOINT FOR TESTING")
    print("=" * 70)
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading checkpoint: {best_model_path}")
    model = LightningModuleSingingVoice.load_from_checkpoint(best_model_path, weights_only=True)

    print("\n" + "=" * 70)
    print("STARTING TESTING")
    print("=" * 70)
    trainer.test(model, test_loader)

    if logger is not None:
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            metrics[key] = value.item() if hasattr(value, "item") else value
        if args.logger == "wandb":
            logger.log_hyperparams(vars(args))
        else:
            logger.log_hyperparams(vars(args), metrics)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {args.eval_metric}: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
