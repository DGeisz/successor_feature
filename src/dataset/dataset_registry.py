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
    SecondNameCorruptedDataset,
    PrefixNamesCorruptedDataset,
    # FirstNameCorruptedDataset,
    LastNumberCorruptedDataset,
    SecondNumberCorruptedDataset,
    # FirstNumberCorruptedDataset,
]

selected_dataset_cache = dict()
first_run_dataset_cache = dict()


def get_corrupted_datasets(N: int, first_run=False):
    if first_run:
        dataset_cache = first_run_dataset_cache
    else:
        dataset_cache = selected_dataset_cache

    if N in dataset_cache:
        return dataset_cache[N]

    corrupted_datasets = [D(N, first_run=first_run) for D in CORRUPTED_DATASET_CLASSES]
    dataset_cache[N] = corrupted_datasets

    return corrupted_datasets
