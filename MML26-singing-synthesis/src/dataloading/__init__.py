from .klangio_dataloader import KlangioDataset
from .synthetic_dataset import SyntheticDataset

DATASET_REGISTRY = {
    "Klangio": {
        "class": KlangioDataset,
        "onset_tolerance": 0.075,
    },
    "Synthetic": {
        "class": SyntheticDataset,
        "onset_tolerance": 0.05,
    },
}


def get_dataset_registry() -> dict:
    """Return the dataset registry."""
    return DATASET_REGISTRY
