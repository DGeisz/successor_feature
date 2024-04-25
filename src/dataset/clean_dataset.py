import random

from src.dataset.dataset import Dataset, BASIC_TEMPLATE
from src.dataset.raw_inputs import US_STATES_SINGLE_WORD


class CleanDataset(Dataset):
    name = "Clean"

    def generate_prompt_strings(self):
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 3)

            candidate = BASIC_TEMPLATE.format(*random_states)

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset


dataset_cache = dict()


def get_clean_dataset(N: int):
    if N in dataset_cache:
        return dataset_cache[N]

    clean_dataset = CleanDataset(N, first_set=True)
    dataset_cache[N] = clean_dataset

    return clean_dataset
