import random

from src.dataset.dataset import Dataset, BASIC_TEMPLATE
from src.dataset.raw_inputs import US_STATES_SINGLE_WORD, MALE_NAMES, BASIC_FILLER_WORDS
from typing import List


class LastNameCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 2)
            random_name = random.choice(MALE_NAMES)

            candidate = BASIC_TEMPLATE.format(
                random_states[0],
                random_states[1],
                random_name,
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset


class SecondNameCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 2)
            random_filler = random.choice(BASIC_FILLER_WORDS)

            candidate = BASIC_TEMPLATE.format(
                random_states[0],
                random_filler,
                random_states[1],
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset


class PrefixNamesCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_state = random.choice(US_STATES_SINGLE_WORD)
            random_fillers = random.sample(BASIC_FILLER_WORDS, 2)

            candidate = BASIC_TEMPLATE.format(
                random_fillers[0],
                random_fillers[1],
                random_state,
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset
