import random

from src.dataset.dataset import Dataset, BASIC_TEMPLATE
from src.dataset.raw_inputs import US_STATES_SINGLE_WORD, MALE_NAMES, BASIC_FILLER_WORDS
from typing import List

LAST_NUMBER_CORRUPTED_TEMPLATE = "14. {} 15. {} {}. {}"


class LastNumberCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 3)
            random_name = random.choice(MALE_NAMES)

            random_number = random.randint(10, 30)

            if random_number in [16, 17, 18]:
                continue

            candidate = BASIC_TEMPLATE.format(
                random_states[0],
                random_states[1],
                random_number,
                random_states[2],
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset


SECOND_NUMBER_CORRUPTED_TEMPLATE = "14. {} {}. {} 16. {}"


class SecondNumberCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 3)

            random_number = random.randint(17, 25)

            candidate = BASIC_TEMPLATE.format(
                random_states[0],
                random_number,
                random_states[1],
                random_states[2],
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset


FIRST_NUMBER_CORRUPTED_TEMPLATE = "{}. {} 15. {} 16. {}"


class FirstNumberCorruptedDataset(Dataset):
    def generate_prompt_strings(self) -> List[str]:
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 3)

            random_number = random.randint(17, 25)

            candidate = BASIC_TEMPLATE.format(
                random_number,
                random_states[0],
                random_states[1],
                random_states[2],
            )

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset
