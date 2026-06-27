import re
import warnings
from pathlib import Path

from ..constants import SAMPLE_RATE
from torch.utils.data import Dataset
import torchaudio


class InferenceDataset(Dataset):
    """
    Dataset for loading audio files for inference.
    
    Args:
        path: Path to a directory containing audio files or a single audio file
        include_pattern: Optional regex pattern to include only matching files
        exclude_pattern: Optional regex pattern to exclude matching files
        device: Device to load audio tensors to (default: "cpu")
    """
    
    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def __init__(self, path, include_pattern=None, exclude_pattern=None, device="cpu"):
        self.path = Path(path)
        self.device = device
        self.include_pattern = re.compile(include_pattern) if include_pattern else None
        self.exclude_pattern = re.compile(exclude_pattern) if exclude_pattern else None
        
        if not self.path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.path}")
        
        self.files = self._find_files()
        print(f"Found {len(self.files)} audio file(s) in {self.path}")

    def _find_files(self):
        """Find all valid audio files in the given path."""
        files = []
        
        # Handle single file
        if self.path.is_file():
            if self._is_valid_audio_file(self.path):
                files.append(self.path)
            else:
                print(f"Warning: {self.path} is not a valid audio file")
            return sorted(files)
        
        # Handle directory
        if not self.path.is_dir():
            raise ValueError(f"Path is neither a file nor a directory: {self.path}")
        
        # Search for audio files recursively
        for file_path in self.path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Apply include/exclude patterns
            if self.include_pattern and not self.include_pattern.search(str(file_path)):
                continue
            if self.exclude_pattern and self.exclude_pattern.search(str(file_path)):
                continue
            
            # Check if it's a valid audio file
            if self._is_valid_audio_file(file_path):
                files.append(file_path)
        
        return sorted(files)
    
    def _is_valid_audio_file(self, file_path: Path) -> bool:
        """Check if a file is a valid audio file that can be loaded."""
        # Check extension first for efficiency
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False
        
        # Try to read metadata
        try:
            torchaudio.info(str(file_path))
            return True
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            # Suppress warning about PySoundFile availability
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
                audio, sr = torchaudio.load(str(file_path), normalize=True)
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(
                    waveform=audio, orig_freq=sr, new_freq=SAMPLE_RATE
                )
            
            # Convert to mono by averaging channels
            audio = audio.mean(dim=0, keepdim=True)
            
            return {"audio": audio.to(self.device), "file_path": str(file_path)}
        
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {file_path}: {e}") from e
