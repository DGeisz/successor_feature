from src.dataset.name_corrupted_datasets import (
    FirstNameCorruptedDataset,
    LastNameCorruptedDataset,
    SecondNameCorruptedDataset,
    PrefixNamesCorruptedDataset,
)
from src.dataset.number_corrupted_dataset import (
    LastNumberCorruptedDataset,
    SecondNumberCorruptedDataset,
    FirstNumberCorruptedDataset,
)


CORRUPTED_DATASET_CLASSES = [
    LastNameCorruptedDataset,
    # SecondNameCorruptedDataset,
    PrefixNamesCorruptedDataset,
    # FirstNameCorruptedDataset,
    LastNumberCorruptedDataset,
    SecondNumberCorruptedDataset,
    # FirstNumberCorruptedDataset,
]

dataset_cache = dict()


def get_corrupted_datasets(N: int):
    if N in dataset_cache:
        return dataset_cache[N]

    corrupted_datasets = [D(N) for D in CORRUPTED_DATASET_CLASSES]
    dataset_cache[N] = corrupted_datasets

    return corrupted_datasets
