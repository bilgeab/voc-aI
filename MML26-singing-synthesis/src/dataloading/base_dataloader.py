import os
import warnings
from abc import abstractmethod
from pathlib import Path

import numpy as np
import soundfile
import torchaudio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constants import SAMPLE_RATE, HOP_LENGTH, ACTIVE_FRAME_LABEL, ONSET_LABEL

import pandas as pd

from typing import NamedTuple

class AudioMetadata(NamedTuple):
    sample_rate: int
    num_frames: int
    num_channels: int

MAX_FLOAT16_VALUE = torch.finfo(torch.float16).max
MAX_FLOAT32_VALUE = torch.finfo(torch.float32).max
MAX_VELOCITY = 128.0


class PianoRollAudioDataset(Dataset):
    def __init__(
        self,
        path,
        groups=None,
        sequence_length=None,
        seed=42,
        device="cpu",
        num_workers=4,
        use_chunks_only_with_onsets=True,
    ):
        self.path = Path(path)
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.num_workers = num_workers
        self.use_chunks_only_with_onsets = use_chunks_only_with_onsets

        self.data = []
        self.chunk_indices = []
        self._tsv_cache = {}

        print(
            f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
            f"of {self.__class__.__name__} at {path}"
        )

        for group in groups:
            for input_files in tqdm(self.files(group), desc=f"loading group {group}"):
                audio_path, tsv_path = input_files
                full_audio_length = self.get_audio_length(audio_path)

                if self.sequence_length is not None:
                    chunk_starts = set()
                    if full_audio_length <= self.sequence_length:
                        chunk_starts.add(0)
                    else:
                        for chunk_start in range(
                            0, full_audio_length, self.sequence_length
                        ):
                            chunk_starts.add(
                                min(
                                    chunk_start,
                                    full_audio_length - self.sequence_length,
                                )
                            )
                        chunk_starts.add(full_audio_length - self.sequence_length)

                    for chunk_start in sorted(chunk_starts):
                        label = self._load_labels_chunk(
                            tsv_path, chunk_start, self.sequence_length
                        )
                        chunk_contains_onset = torch.any(label == ONSET_LABEL)
                        if not self.use_chunks_only_with_onsets or chunk_contains_onset:
                            self.data.append({"path": audio_path, "tsv_path": tsv_path})
                            self.chunk_indices.append((len(self.data) - 1, chunk_start))
                else:
                    self.data.append({"path": audio_path, "tsv_path": tsv_path})
                    self.chunk_indices.append((len(self.data) - 1, 0))

        print(f"Loaded {len(self.chunk_indices)} chunks and {len(self.data)} files")

    def __getitem__(self, index):
        data_index, chunk_start = self.chunk_indices[index]
        data = self.data[data_index]

        result = dict(song_id=Path(data["path"]).parent.name, dataset=self.__str__())

        if self.sequence_length is not None:
            audio = self._load_audio(
                data["path"], offset=chunk_start, num_frames=self.sequence_length
            )
            if audio is None:
                raise RuntimeError(f"Error loading data chunk for {data['path']}")
            if audio.shape[1] < self.sequence_length:
                pad_length = self.sequence_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, pad_length))

            label = self._load_labels_chunk(
                data["tsv_path"], chunk_start, audio.shape[1]
            )
            expected_frames = (audio.shape[1] - 1) // HOP_LENGTH + 1
            if label.shape[0] < expected_frames:
                label = torch.nn.functional.pad(
                    label, (0, 0, 0, expected_frames - label.shape[0])
                )

            result["audio"] = audio.to(self.device)
            score_label = label.to(self.device)
        else:
            audio = self._load_audio(data["path"])
            if audio is None:
                raise RuntimeError(f"Error loading audio for {data['path']}")
            full_audio_length = audio.shape[1]
            label, midi = self._load_labels(data["tsv_path"], full_audio_length)
            if label is None:
                raise RuntimeError(f"Error loading labels for {data['path']}")

            result["audio"] = audio.to(self.device)
            result["notes"] = midi
            score_label = label.to(self.device)

        result["onset"] = (score_label == ONSET_LABEL).short()
        result["frame"] = (score_label >= ACTIVE_FRAME_LABEL).short()

        return result

    def __len__(self):
        return len(self.chunk_indices)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def _load_tsv_cached(self, tsv_path):
        tsv_key = str(tsv_path)
        if tsv_key not in self._tsv_cache:
            midi = np.loadtxt(tsv_path, delimiter="\t", skiprows=1)
            if midi.ndim == 1:
                midi = midi[np.newaxis, :]
            self._tsv_cache[tsv_key] = midi
        return self._tsv_cache[tsv_key]
        
    @staticmethod
    def _get_audio_metadata(audio_path):
        path = str(audio_path)
        try:
            info = torchaudio.info(path)
            return AudioMetadata(
                sample_rate=info.sample_rate,
                num_frames=info.num_frames,
                num_channels=info.num_channels,
            )
        except (AttributeError, Exception):
            sf = soundfile.info(path)
            return AudioMetadata(
                sample_rate=sf.samplerate,
                num_frames=sf.frames,
                num_channels=sf.channels,
            )

    def _load_audio(self, audio_path, offset=0, num_frames=-1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
            if num_frames > 0:
                info = self._get_audio_metadata(audio_path)
                file_sr = info.sample_rate
                offset_src = int(round(offset * file_sr / SAMPLE_RATE))
                num_frames_src = int(round(num_frames * file_sr / SAMPLE_RATE))
                audio, sr = torchaudio.load(
                    str(audio_path),
                    frame_offset=offset_src,
                    num_frames=num_frames_src,
                    normalize=True,
                )
            else:
                audio, sr = torchaudio.load(str(audio_path), normalize=True)

        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(
                waveform=audio, orig_freq=sr, new_freq=SAMPLE_RATE
            )

        audio = audio.mean(0, keepdim=True)

        if num_frames > 0:
            if audio.shape[1] > num_frames:
                audio = audio[:, :num_frames]
            elif audio.shape[1] < num_frames:
                audio = torch.nn.functional.pad(
                    audio, (0, num_frames - audio.shape[1])
                )

        return audio

    def get_audio_length(self, audio_path):
        """
        Return the lenght of the audio in samples (adjusted to the project sample rate)
        """
        info = self._get_audio_metadata(audio_path)
        return int(info.num_frames * SAMPLE_RATE / info.sample_rate)

    def _load_labels(self, tsv_path, audio_length, chunk_start=None):
        n_keys = 127
        midi = self._load_tsv_cached(tsv_path)

        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        chunk_start_seconds = chunk_start / SAMPLE_RATE if chunk_start is not None else 0.0
        chunk_duration_s = audio_length / SAMPLE_RATE

        for onset, offset, note in midi:
            onset -= chunk_start_seconds
            offset -= chunk_start_seconds

            if chunk_start is not None:
                if offset < 0 or onset >= chunk_duration_s:
                    continue

            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + 1)
            frame_left = max(0, onset_right)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)

            f = int(note)
            if left >= 0:
                label[left:onset_right, f] = ONSET_LABEL
            label[frame_left:frame_right, f] = ACTIVE_FRAME_LABEL

        if chunk_start is None:
            return label, torch.from_numpy(midi).float()
        return label

    def _load_labels_chunk(self, tsv_path, chunk_start, sequence_length):
        return self._load_labels(tsv_path, sequence_length, chunk_start=chunk_start)
