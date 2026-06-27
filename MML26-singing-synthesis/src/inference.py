"""
Inference script for singing voice transcription.

Usage:
    python -m src.inference --checkpoint-path <ckpt> --input-path <audio_dir> --output-dir <output_dir>

Run `python -m src.inference --help` for all options.
"""

import argparse
import json
from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .lightning_module import LightningModuleSingingVoice
from .dataloading.inference_dataset import InferenceDataset

# Optional MIDI export support
try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on audio files for singing voice transcription",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint to use for inference",
    )
    # Input/Output arguments
    io_group = parser.add_argument_group("Input/Output Configuration")
    io_group.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to directory containing audio files or a single audio file",
    )
    io_group.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory to save prediction results",
    )
    io_group.add_argument(
        "--include-pattern",
        type=str,
        default=None,
        help="Regex pattern to include only matching files (e.g., '.*\\.wav$')",
    )
    io_group.add_argument(
        "--exclude-pattern",
        type=str,
        default=None,
        help="Regex pattern to exclude matching files",
    )
    io_group.add_argument(
        "--output-format",
        type=str,
        default="txt",
        choices=["txt", "csv", "json", "midi", "all"],
        help="Output format for predictions (use 'midi' or --export-as-midi for MIDI files)",
    )
    io_group.add_argument(
        "--export-as-midi",
        action="store_true",
        help="Export predictions as MIDI files (requires pretty_midi library)",
    )
    io_group.add_argument(
        "--save-activations",
        action="store_true",
        help="Save model activations (onset/frame probabilities) as .npz files",
    )

    # System arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "mps"],
        help="Hardware accelerator to use",
    )
    system_group.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    system_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    system_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (typically 1 for variable-length audio)",
    )

    return parser.parse_args()


def save_notes_txt(notes: np.ndarray, output_path: Path) -> None:
    """Save notes in tab-separated text format (onset, offset, pitch)."""
    with open(output_path, "w") as f:
        f.write("onset\toffset\tpitch\n")
        for note in notes:
            f.write(f"{note[0]:.6f}\t{note[1]:.6f}\t{note[2]:.0f}\n")


def save_notes_csv(notes: np.ndarray, output_path: Path) -> None:
    """Save notes in CSV format."""
    with open(output_path, "w") as f:
        f.write("onset,offset,pitch\n")
        for note in notes:
            f.write(f"{note[0]:.6f},{note[1]:.6f},{note[2]:.0f}\n")


def save_notes_json(notes: np.ndarray, output_path: Path) -> None:
    """Save notes in JSON format."""
    notes_list = [
        {
            "onset": float(note[0]),
            "offset": float(note[1]),
            "pitch": int(note[2])
        }
        for note in notes
    ]
    with open(output_path, "w") as f:
        json.dump(notes_list, f, indent=2)


def save_notes_midi(notes: np.ndarray, output_path: Path, tempo: float = 120.0) -> None:
    """Save notes as MIDI file.
    
    Args:
        notes: Numpy array of shape (N, 3) with columns [onset, offset, pitch]
        output_path: Path to save the MIDI file
        tempo: Tempo in BPM (default: 120)
    """
    if not PRETTY_MIDI_AVAILABLE:
        raise ImportError(
            "pretty_midi is required for MIDI export. Install with: pip install pretty_midi"
        )
    
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create an instrument (Vocal/Voice)
    instrument = pretty_midi.Instrument(program=53)  # Voice Oohs (MIDI program 53)
    
    # Add notes to the instrument
    for onset, offset, pitch in notes:
        # Ensure valid duration
        if offset <= onset:
            offset = onset + 0.05  # Minimum 50ms duration
        
        # Clamp pitch to valid MIDI range (0-127)
        pitch = int(np.clip(pitch, 0, 127))
        
        # Create note with default velocity
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=float(onset),
            end=float(offset)
        )
        instrument.notes.append(note)
    
    # Add the instrument to the MIDI object
    midi.instruments.append(instrument)
    
    # Write to file
    midi.write(str(output_path))


def save_activations(activations: dict, output_path: Path) -> None:
    """Save model activations as compressed numpy file."""
    np.savez_compressed(output_path, **activations)


def get_output_path(input_file_path: str, input_base_path: str, output_dir: Path, suffix: str) -> Path:
    """
    Generate output path based on input file path, preserving directory structure.
    
    Args:
        input_file_path: Full path to the input file
        input_base_path: Base input directory or file path
        output_dir: Output directory root
        suffix: File suffix to append (e.g., "_notes.txt", ".mid")
    
    Returns:
        Path object for the output file with preserved directory structure
    """
    input_file = Path(input_file_path).resolve()
    input_base = Path(input_base_path).resolve()
    
    # If input_base is a file, use its parent directory as base
    if input_base.is_file():
        input_base = input_base.parent
        # For single file, just use the filename
        relative_path = input_file.parent.name / input_file.stem
    else:
        # For directories, compute relative path from base to file
        try:
            relative_path = input_file.relative_to(input_base).parent / input_file.stem
        except ValueError:
            # If file is not relative to base (shouldn't happen), fall back to just filename
            relative_path = Path(input_file.stem)
    
    # Create output path preserving directory structure
    output_path = output_dir / relative_path.parent / f"{relative_path.name}{suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path


def main() -> None:
    args = parse_args()

    # Validate MIDI export requirements
    if args.export_as_midi or args.output_format in ["midi", "all"]:
        if not PRETTY_MIDI_AVAILABLE:
            print("ERROR: MIDI export requested but pretty_midi is not installed.")
            print("Install with: pip install pretty_midi")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INFERENCE CONFIGURATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Input Path: {args.input_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Output Format: {args.output_format}")
    print(f"Export as MIDI: {args.export_as_midi or args.output_format in ['midi', 'all']}")
    print(f"Save Activations: {args.save_activations}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Devices: {args.devices}")
    print("=" * 70)
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    model = LightningModuleSingingVoice.load_from_checkpoint(
        args.checkpoint_path,
        weights_only=True,
    )
    print(f"Model loaded (OAF / Basic Pitch)")
    print(f"Thresholds - Onset: {model.onset_threshold:.3f}, "
          f"Frame: {model.frame_threshold:.3f}")
    print()

    # Setup dataset
    print("Loading audio files...")
    device = "cpu"  # Load audio on CPU, model will handle device placement
    dataset = InferenceDataset(
        path=args.input_path,
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
        device=device,
    )

    if len(dataset) == 0:
        print("No audio files found! Check your input path and patterns.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Run inference
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    predictions = trainer.predict(model, dataloader)

    # Process and save predictions
    print("\n" + "=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)
    
    # Store input base path for preserving directory structure
    input_base_path = args.input_path
    
    for estimated_notes, file_path, activations in predictions:
        print(f"Processing: {file_path}")
        print(f"  Found {len(estimated_notes)} notes")
        
        # Save notes in requested format(s)
        if args.output_format in ["txt", "all"]:
            output_path = get_output_path(file_path, input_base_path, output_dir, "_notes.txt")
            save_notes_txt(estimated_notes, output_path)
            print(f"  Saved TXT: {output_path}")
        
        if args.output_format in ["csv", "all"]:
            output_path = get_output_path(file_path, input_base_path, output_dir, "_notes.csv")
            save_notes_csv(estimated_notes, output_path)
            print(f"  Saved CSV: {output_path}")
        
        if args.output_format in ["json", "all"]:
            output_path = get_output_path(file_path, input_base_path, output_dir, "_notes.json")
            save_notes_json(estimated_notes, output_path)
            print(f"  Saved JSON: {output_path}")
        
        if args.export_as_midi or args.output_format in ["midi", "all"]:
            output_path = get_output_path(file_path, input_base_path, output_dir, ".mid")
            save_notes_midi(estimated_notes, output_path)
            print(f"  Saved MIDI: {output_path}")
        
        # Save activations if requested
        if args.save_activations:
            output_path = get_output_path(file_path, input_base_path, output_dir, "_activations.npz")
            save_activations(activations, output_path)
            print(f"  Saved activations: {output_path}")
    
    print("\n" + "=" * 70)
    print(f"INFERENCE COMPLETE")
    print(f"Processed {len(predictions)} files")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
