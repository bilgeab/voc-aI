from typing import Dict, Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

from .constants import (
    SAMPLE_RATE,
    HOP_LENGTH,
    MIN_NOTE_LEN_FRAMES,
    DEBUG_PRINT_FRAME_AND_ONSET_DURING_EVALUATION,
    BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE,
    BASIC_PITCH_N_CONTOUR_BINS,
    BASIC_PITCH_N_SEMITONES,
    PIANO_MIDI_START,
)
from .dataloading import get_dataset_registry
from .basic_pitch_transcriber import BasicPitchTranscriber
from .transcription_utils import (
    output_to_notes_polyphonic,
    get_metrics_dict,
)
from mir_eval.util import match_events


class LightningModuleSingingVoice(pl.LightningModule):
    def __init__(
            self,
            learning_rate=1e-3,
            optimizer_type="adam",
            onset_weight: float = 1.0,
            frame_weight: float = 1.0,
            chunk_duration_s: float = 8.0,
            context_duration_s: float = 2.0,
            batch_size_inference: int = 8,
            # Legacy checkpoint compatibility (ignored)
            use_key_estimation: bool | str = False,
            model: str | None = None,
            prediction_type: str | None = None,
            cover_loss_weight: float = 0.0,
            cover_loss_type: str = "cosine",
    ):
        super().__init__()

        self.save_hyperparameters(
            "learning_rate",
            "optimizer_type",
            "onset_weight",
            "frame_weight",
            "chunk_duration_s",
            "context_duration_s",
            "batch_size_inference",
        )
        self.examples_validation = []
        self.examples_test = []
        self.model = BasicPitchTranscriber()
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.onset_threshold = 0.4
        self.frame_threshold = 0.5
        self.vad_threshold = 0.5
        self.chunk_duration_s = chunk_duration_s
        self.context_duration_s = context_duration_s
        self.batch_size_inference = batch_size_inference
        self.prediction_type = "oaf"

        self.frame_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(frame_weight)
        )
        self.contour_loss = nn.BCEWithLogitsLoss()
        self.onset_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(onset_weight)
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["onset_threshold"] = self.onset_threshold
        checkpoint["frame_threshold"] = self.frame_threshold
        checkpoint["vad_threshold"] = self.vad_threshold

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.onset_threshold = checkpoint.get("onset_threshold", 0.4)
        self.frame_threshold = checkpoint.get("frame_threshold", 0.5)
        self.vad_threshold = checkpoint.get("vad_threshold", 0.5)

    def _get_onset_tolerance(self, dataset_name: str) -> float:
        registry = get_dataset_registry()
        return registry.get(dataset_name, {}).get("onset_tolerance", 0.05)

    def _log_figure(self, name: str, fig) -> None:
        if self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(name, fig, global_step=self.global_step)
            return
        try:
            import wandb
        except ImportError:
            return
        try:
            experiment.log({name: wandb.Image(fig)}, step=self.global_step)
        except Exception:
            return

    def _plot_oaf_validation(self, example):
        frames = example["predicted_frames"].T
        onsets = example["predicted_onsets"].T
        target_frames = example["target_frames"].T
        target_onsets = example["target_onsets"].T

        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle(f'OAF Validation Example: {example["song_id"]}')
        ax[0][0].imshow(frames, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[0][0].set_title("Predicted frames")
        ax[0][1].imshow(onsets, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[0][1].set_title("Predicted onsets")
        ax[1][0].imshow(target_frames, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[1][0].set_title("Target frames")
        ax[1][1].imshow(target_onsets, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1)
        ax[1][1].set_title("Target onsets")
        return fig

    def _plot_pitch_error_histogram(self, examples, onset_tolerance=0.05):
        all_pitch_errors = []

        for example in examples:
            reference_notes = example["notes"]
            frames = example["predicted_frames"]
            onsets = example["predicted_onsets"]
            predicted_notes = output_to_notes_polyphonic(
                frames,
                onsets,
                onset_thresh=self.onset_threshold,
                frame_thresh=self.frame_threshold,
                min_note_len=MIN_NOTE_LEN_FRAMES,
                infer_onsets=True,
            )

            if len(predicted_notes) > 0 and len(reference_notes) > 0:
                ref_onsets = reference_notes[:, 0]
                est_onsets = predicted_notes[:, 0]
                matching = np.array(match_events(ref_onsets, est_onsets, onset_tolerance))

                if len(matching) > 0:
                    ref_matched_pitches = reference_notes[matching[:, 0], 2]
                    est_matched_pitches = predicted_notes[matching[:, 1], 2]
                    pitch_errors = est_matched_pitches - ref_matched_pitches
                    all_pitch_errors.extend(pitch_errors)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if len(all_pitch_errors) > 0:
            pitch_errors_array = np.array(all_pitch_errors)
            ax.hist(
                pitch_errors_array,
                bins=np.arange(-12.5, 13.5, 1),
                edgecolor='black',
                alpha=0.7,
                color='steelblue'
            )
            mean_error = np.mean(pitch_errors_array)
            std_error = np.std(pitch_errors_array)
            median_error = np.median(pitch_errors_array)
            ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
            ax.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_xlabel('Pitch Error (semitones)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(
                f'Pitch Error Histogram (OAF)\n'
                f'N={len(pitch_errors_array)} matched notes | '
                f'Mean={mean_error:.2f} | Std={std_error:.2f} | Median={median_error:.2f}',
                fontsize=13
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(-12, 13, 2))
        else:
            ax.text(0.5, 0.5, 'No matched notes',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Pitch Error Histogram (No Data)')

        fig.tight_layout()
        return fig

    def forward(self, x, return_contour: bool = False):
        return self.model(x, return_contour=return_contour)

    def _context_audio_samples(self) -> int:
        return int(SAMPLE_RATE * self.context_duration_s)

    def _context_frame_count(self) -> int:
        context_audio = self._context_audio_samples()
        return (context_audio - 1) // HOP_LENGTH + 1

    def _forward_with_context(
        self, audio: torch.Tensor, return_contour: bool = False
    ) -> dict[str, torch.Tensor]:
        """Run the model with edge context; return logits for the center region only."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        context_audio = self._context_audio_samples()
        context_frames = self._context_frame_count()
        center_samples = audio.shape[-1]
        center_frames = (center_samples - 1) // HOP_LENGTH + 1

        padded = F.pad(audio, (context_audio, context_audio))
        outputs = self.model(padded, return_contour=return_contour)

        stripped: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if value.dim() >= 2 and value.shape[1] >= 2 * context_frames:
                stripped[key] = value[
                    :, context_frames : context_frames + center_frames
                ]
            else:
                stripped[key] = value[:, :center_frames]
        return stripped

    def _decode_notes(
        self,
        onset_logits: torch.Tensor,
        frame_logits: torch.Tensor,
        onset_thresh: float | None = None,
        frame_thresh: float | None = None,
    ) -> np.ndarray:
        if onset_thresh is None:
            onset_thresh = self.onset_threshold
        if frame_thresh is None:
            frame_thresh = self.frame_threshold

        onsets = torch.sigmoid(onset_logits).detach().cpu().numpy()
        frames = torch.sigmoid(frame_logits).detach().cpu().numpy()
        return output_to_notes_polyphonic(
            frames=frames,
            onsets=onsets,
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
            min_note_len=MIN_NOTE_LEN_FRAMES,
            infer_onsets=True,
        )

    def convert_labels(self, onset, frame):
        return {
            "onset": onset.clamp(0, 1).float(),
            "frame": frame.clamp(0, 1).float(),
        }

    def _build_contour_targets(
        self,
        frame_labels: torch.Tensor,
        n_contour_frames: int,
    ) -> torch.Tensor:
        """Build Basic Pitch contour targets from frame labels (training only).

        Expands 88 piano semitones to 3 bins/semitone on the shared frame grid.
        """
        piano_end = PIANO_MIDI_START + BASIC_PITCH_N_SEMITONES
        piano_frames = frame_labels[..., PIANO_MIDI_START:piano_end].float()
        contour = piano_frames.repeat_interleave(
            BASIC_PITCH_CONTOURS_BINS_PER_SEMITONE, dim=-1
        )
        assert contour.shape[-1] == BASIC_PITCH_N_CONTOUR_BINS

        if contour.shape[1] != n_contour_frames:
            contour = F.interpolate(
                contour.transpose(1, 2),
                size=n_contour_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return contour.clamp(0, 1)

    def _bce_fixed_batch_denominator(self, logits, targets, pos_weight=None):
        batch_size = logits.shape[0]
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        sample_losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight,
        ).reshape(batch_size, -1).mean(dim=1)
        return sample_losses.sum() / batch_size

    def compute_losses(self, model_outputs, labels):
        onset_out = model_outputs["onset"]
        frame_out = model_outputs["frame"]
        onset_label = labels["onset"]
        frame_label = labels["frame"]

        onset_pos_weight = self.onset_loss.pos_weight
        if onset_pos_weight is not None:
            onset_pos_weight = onset_pos_weight.to(onset_out.device)
        onset_loss = self._bce_fixed_batch_denominator(
            onset_out, onset_label, pos_weight=onset_pos_weight,
        )
        frame_pos_weight = self.frame_loss.pos_weight
        if frame_pos_weight is not None:
            frame_pos_weight = frame_pos_weight.to(frame_out.device)
        frame_loss = self._bce_fixed_batch_denominator(
            frame_out, frame_label, pos_weight=frame_pos_weight,
        )
        return {"onset": onset_loss, "frame": frame_loss}

    def compute_contour_loss(
        self,
        contour_logits: torch.Tensor,
        frame_labels: torch.Tensor,
    ) -> torch.Tensor:
        contour_targets = self._build_contour_targets(
            frame_labels, contour_logits.shape[1]
        )
        return self._bce_fixed_batch_denominator(
            contour_logits,
            contour_targets,
        )

    def training_step(self, batch, batch_idx):
        converted_labels = self.convert_labels(batch["onset"], batch["frame"])
        model_outputs = self._forward_with_context(
            batch["audio"], return_contour=True
        )
        loss_dict = self.compute_losses(model_outputs, converted_labels)
        contour_loss = self.compute_contour_loss(
            model_outputs["contour"],
            converted_labels["frame"],
        )
        loss_dict["contour"] = contour_loss
        for loss_key, loss_value in loss_dict.items():
            self.log(f"train/{loss_key}", loss_value, prog_bar=True, on_step=True, on_epoch=False)
        return sum(loss_dict.values())

    def on_validation_epoch_start(self) -> None:
        self.examples_validation = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        converted_labels = self.convert_labels(batch["onset"], batch["frame"])
        example_data = {
            "song_id": batch["song_id"][0],
            "dataset": batch["dataset"][0],
            "notes": batch["notes"][0, ...].detach().cpu().numpy(),
        }
        model_outputs = self._framewise_inference_on_batches(batch["audio"])
        loss_dict = self.compute_losses(model_outputs, converted_labels)
        for loss_key, loss_val in loss_dict.items():
            self.log(f"eval/{loss_key}", loss_val, prog_bar=True, batch_size=batch["audio"].shape[0])

        assert batch["audio"].shape[0] == 1, "For evaluation, batch size must be of size one"

        example_data.update({
            "predicted_onsets": torch.sigmoid(model_outputs["onset"][0, ...]).detach().cpu().numpy(),
            "predicted_frames": torch.sigmoid(model_outputs["frame"][0, ...]).detach().cpu().numpy(),
            "target_onsets": converted_labels["onset"][0, ...].detach().cpu().numpy(),
            "target_frames": converted_labels["frame"][0, ...].detach().cpu().numpy(),
        })
        self.examples_validation.append(example_data)
        return sum(loss_dict.values())

    def on_validation_epoch_end(self):
        if not self.examples_validation:
            print("Warning: No validation examples recorded.")
            return

        results = []
        for onset_thresh in np.linspace(0.1, 0.9, 3):  # Wow! You are really curious about the code!
            for frame_thresh in np.linspace(0.1, 0.9, 3):  # Do not be shy to mess around these values
                for example in self.examples_validation:     # But mind that the more threshold combinations that you explore
                    predicted_notes = output_to_notes_polyphonic( # The more time each validation epoch will take
                        example["predicted_frames"],
                        example["predicted_onsets"],
                        onset_thresh=onset_thresh,
                        frame_thresh=frame_thresh,
                        min_note_len=MIN_NOTE_LEN_FRAMES,
                        infer_onsets=True,
                    )
                    onset_tolerance = self._get_onset_tolerance(example["dataset"])
                    metrics = get_metrics_dict(
                        ref_notes=example["notes"],
                        est_notes=predicted_notes,
                        onset_tolerance=onset_tolerance,
                    )
                    results.append({
                        "song_id": example["song_id"],
                        "dataset": example["dataset"],
                        "onset_thresh": onset_thresh,
                        "frame_thresh": frame_thresh,
                        **metrics,
                    })

        if not results:
            print("Warning: No validation results generated during threshold optimization.")
            self.examples_validation = []
            return

        results_df = pd.DataFrame(results)
        results_df_grouped = (
            results_df.groupby(["onset_thresh", "frame_thresh"])
            .mean(numeric_only=True)
            .reset_index()
        )
        best_metrics_row = results_df_grouped.loc[results_df_grouped["COnP_f1"].idxmax()]
        self.onset_threshold = best_metrics_row["onset_thresh"]
        self.frame_threshold = best_metrics_row["frame_thresh"]
        best_metrics_agg = best_metrics_row.to_dict()
        print(
            f"Best OAF thresholds: Onset={self.onset_threshold:.2f}, "
            f"Frame={self.frame_threshold:.2f}"
        )

        for metric_key, val in best_metrics_agg.items():
            if metric_key not in ["onset_thresh", "frame_thresh"]:
                self.log(f"eval/{metric_key}", val, prog_bar=True)

        for dataset_name in results_df["dataset"].unique():
            dataset_results = results_df[
                (results_df["dataset"] == dataset_name)
                & (results_df["onset_thresh"] == self.onset_threshold)
                & (results_df["frame_thresh"] == self.frame_threshold)
            ]
            if len(dataset_results) > 0:
                dataset_metrics = dataset_results.select_dtypes(include=[np.number]).mean()
                for metric_key, val in dataset_metrics.items():
                    if metric_key not in ["onset_thresh", "frame_thresh"]:
                        self.log(f"eval/{dataset_name}/{metric_key}", float(val), prog_bar=False)

        self.log("eval/best_onset_thresh", self.onset_threshold, prog_bar=False)
        self.log("eval/best_frame_thresh", self.frame_threshold, prog_bar=False)

        fig_hist = self._plot_pitch_error_histogram(self.examples_validation)
        self._log_figure("OAF Pitch Error Histogram (Best Thresholds)", fig_hist)
        plt.close(fig_hist)

        if DEBUG_PRINT_FRAME_AND_ONSET_DURING_EVALUATION and self.examples_validation:
            fig = self._plot_oaf_validation(self.examples_validation[0])
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._log_figure("OAF Validation Plot", fig)
            plt.close(fig)

        self.examples_validation = []

    def on_test_epoch_start(self) -> None:
        self.examples_test = []

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        model_outputs = self._framewise_inference_on_batches(batch["audio"])
        assert model_outputs["onset"].shape[0] == 1, "For testing, batch size must be of size one"

        reference_notes = batch["notes"][0, ...].detach().cpu().numpy()
        dataset = batch["dataset"][0]

        estimated_notes = self._decode_notes(
            model_outputs["onset"][0, ...],
            model_outputs["frame"][0, ...],
        )

        onset_tolerance = self._get_onset_tolerance(dataset)
        metrics_for_song = get_metrics_dict(
            ref_notes=reference_notes,
            est_notes=estimated_notes,
            onset_tolerance=onset_tolerance,
        )
        self.examples_test.append({
            "song_id": batch["song_id"][0],
            "dataset": dataset,
            **metrics_for_song,
        })

    def on_test_epoch_end(self):
        if not self.examples_test:
            print("Warning: No test examples recorded.")
            return

        results_df = pd.DataFrame(self.examples_test)
        hp_metrics = {}
        for dataset_name in results_df["dataset"].unique():
            dataset_results = results_df[results_df["dataset"] == dataset_name]
            dataset_metrics = dataset_results.select_dtypes(include=[np.number]).mean()
            for metric_key, val in dataset_metrics.items():
                metric_name = f"test/{dataset_name}/{metric_key}"
                self.log(metric_name, val, prog_bar=False, logger=True)
                hp_metrics[metric_name] = val

        global_metrics = results_df.select_dtypes(include=[np.number]).mean()
        for metric_key, val in global_metrics.items():
            metric_name = f"test/global/{metric_key}"
            self.log(metric_name, val, prog_bar=False, logger=True)
            hp_metrics[metric_name] = val

        hp_metrics["test/onset_threshold"] = self.onset_threshold
        hp_metrics["test/frame_threshold"] = self.frame_threshold
        self.log("test/onset_threshold", self.onset_threshold, prog_bar=False)
        self.log("test/frame_threshold", self.frame_threshold, prog_bar=False)

        if self.logger is not None:
            experiment = getattr(self.logger, "experiment", None)
            if experiment is not None:
                if hasattr(experiment, "log"):
                    try:
                        import wandb
                        experiment.log({"hp_metrics": hp_metrics})
                    except Exception as e:
                        print(f"Warning: Could not log hp_metrics to WandB: {e}")
                if hasattr(experiment, "add_hparams"):
                    try:
                        experiment.add_hparams({}, hp_metrics, global_step=self.global_step)
                    except Exception as e:
                        print(f"Warning: Could not log hp_metrics to TensorBoard: {e}")

        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        for dataset_name in sorted(results_df["dataset"].unique()):
            dataset_results = results_df[results_df["dataset"] == dataset_name]
            print(f"\n{dataset_name} ({len(dataset_results)} samples):")
            dataset_metrics = dataset_results.select_dtypes(include=[np.number]).mean()
            for metric_key, val in sorted(dataset_metrics.items()):
                print(f"  {metric_key}: {val:.4f}")
        print("\nGlobal Results (across all datasets):")
        for metric_key, val in sorted(global_metrics.items()):
            print(f"  {metric_key}: {val:.4f}")
        print("=" * 70)
        self.examples_test = []

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        model_outputs = self._framewise_inference_on_batches(batch["audio"])
        assert model_outputs["onset"].shape[0] == 1, "For prediction, batch size must be of size one"

        onset_logits = model_outputs["onset"][0, ...].detach().cpu().numpy()
        frame_logits = model_outputs["frame"][0, ...].detach().cpu().numpy()
        activations = {"onset_logits": onset_logits, "frame_logits": frame_logits}
        estimated_notes = self._decode_notes(
            model_outputs["onset"][0, ...],
            model_outputs["frame"][0, ...],
        )
        return estimated_notes, batch["file_path"][0], activations

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.optimizer_type == "sgd":
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _framewise_inference_on_batches(self, x_batch):
        label_dict = {}

        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)
        assert x_batch.dim() == 3, f"Expected 3D audio input, got {x_batch.dim()}D"
        assert x_batch.shape[0] == 1, "Inference currently assumes batch size of 1"

        chunk_duration_s = self.chunk_duration_s
        context_duration_s = self.context_duration_s
        chunk_length = int(SAMPLE_RATE * chunk_duration_s)
        context_audio = int(SAMPLE_RATE * context_duration_s)
        context_frames = (context_audio - 1) // HOP_LENGTH + 1
        batch_size_inference = self.batch_size_inference

        signal_length = x_batch.shape[-1]
        n_frames_in_audio = (signal_length - 1) // HOP_LENGTH + 1

        chunks = self._create_chunks_with_context(x_batch, chunk_length, context_audio)

        for audio_chunk_batch in chunks.split(batch_size_inference, 0):
            labels = self(audio_chunk_batch.squeeze(1).squeeze(1))
            for key, value in labels.items():
                stripped_value = value[:, context_frames:-context_frames]
                label_dict.setdefault(key, []).append(stripped_value)

        for key, value_list in label_dict.items():
            concatenated_value = torch.cat(value_list, dim=0)
            final_value = concatenated_value.flatten(0, 1).unsqueeze(0)
            label_dict[key] = final_value[:, :n_frames_in_audio, ...]

        return label_dict

    @staticmethod
    def _create_chunks_with_context(signal, Y, C):
        assert signal.dim() == 3, "Signal must have shape (batch, channels, time)"
        assert signal.shape[0] == 1, "_create_chunks_with_context requires batch size 1"

        signal_length = signal.shape[-1]
        chunk_size_with_context = Y + 2 * C
        step_size = Y
        num_strides = (signal_length + C - 1) // step_size
        last_step_start_in_padded = C + num_strides * step_size
        required_padded_length = last_step_start_in_padded + Y + C
        pad_required_end = max(0, required_padded_length - (signal_length + C))

        signal_padded = F.pad(signal, (C, C + pad_required_end))
        chunks = signal_padded.unfold(-1, chunk_size_with_context, step_size)
        return chunks.permute(2, 0, 1, 3)
