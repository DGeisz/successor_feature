import random

from src.dataset.dataset import Dataset, BASIC_TEMPLATE
from src.dataset.raw_inputs import US_STATES_SINGLE_WORD

from typing import List


class CleanDataset(Dataset):
    N: int
    prompt_strings: List[str]

    # def __init__(self, N: int):
    #     self.N = N

    #     self.prompt_strings = self.generate()

    def generate_prompt_strings(self):
        dataset = []

        while len(dataset) < self.N:
            random_states = random.sample(US_STATES_SINGLE_WORD, 3)

            candidate = BASIC_TEMPLATE.format(*random_states)

            if candidate not in dataset:
                dataset.append(candidate)

        dataset = dataset[: self.N]

        return dataset
