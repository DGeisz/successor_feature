import einops

from abc import ABC, abstractmethod
from typing import List
from src.activation import get_linear_feature_activation_from_cache
from src.model import model

BASIC_TEMPLATE = "14. {} 15. {} 16. {}"


class Dataset(ABC):
    N: int
    _cache = None

    def __init__(self, N: int):
        self.N = N

        self.prompt_strings = self.generate_prompt_strings()
        self.tokens = model.to_tokens(self.prompt_strings, prepend_bos=True)

        self.cache

    @abstractmethod
    def generate_prompt_strings(self) -> List[str]:
        pass

    @property
    def cache(self):
        if self._cache == None:
            _, self._cache = model.run_with_cache(self.tokens)

        return self._cache

    def get_average_activation(self):
        return get_linear_feature_activation_from_cache(self.cache, averaged=True)

    def get_mean_patch(self, name: str, batch_size: int):
        return einops.repeat(self.cache[name].mean(dim=0), "... -> n ...", n=batch_size)


def get_normalizing_function_for_datasets(
    clean_dataset: Dataset, corrupted_dataset: Dataset
):
    corrupted_activation = corrupted_dataset.get_average_activation()
    clean_activation = clean_dataset.get_average_activation()

    def normalize_activation(activation):
        return (clean_activation - activation) / (
            clean_activation - corrupted_activation
        )

    return normalize_activation
