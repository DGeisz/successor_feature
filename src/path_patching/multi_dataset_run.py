import torch

from typing import List
from src.dataset.clean_dataset import get_clean_dataset
from src.dataset.dataset import (
    get_corrupted_normalizing_function_for_activations,
    get_mended_normalizing_function_for_activations,
)
from src.dataset.dataset_registry import (
    CORRUPTED_DATASET_CLASSES,
    get_corrupted_datasets,
)
from src.path_patching.single_run import PatchType, SingleRun
from src.plot_utils import imshow
from src.type_utils import AttnHead


class MultiDatasetRun:
    def __init__(
        self,
        N: int,
        target_nodes: List[AttnHead],
        target_location: str,
        patch_type=PatchType.CORRUPT,
        individual_normalization=False,
        first_run=False,
    ):
        self.N = N

        self.clean_dataset = get_clean_dataset(N)
        self.corrupted_datasets = get_corrupted_datasets(N, first_run=first_run)
        # [D(N) for D in CORRUPTED_DATASET_CLASSES]

        min_activation = min(
            d.get_average_activation() for d in self.corrupted_datasets
        )
        clean_activation = self.clean_dataset.get_average_activation()

        if individual_normalization:
            normalizing_function = None
        else:
            if patch_type == PatchType.CORRUPT:
                normalizing_function = (
                    get_corrupted_normalizing_function_for_activations(
                        clean_activation, min_activation
                    )
                )
            else:
                normalizing_function = get_mended_normalizing_function_for_activations(
                    clean_activation, min_activation
                )

        assert target_location in ("q", "k", "v", "z")

        self.target_location = target_location
        self.target_nodes = target_nodes
        self.patch_type = patch_type

        self.individual_runs = [
            SingleRun(
                self.clean_dataset,
                corrupted_dataset,
                target_nodes,
                target_location,
                normalizing_function=normalizing_function,
                patch_type=self.patch_type,
            )
            for corrupted_dataset in self.corrupted_datasets
        ]

    def run(self):
        return [run.run() for run in self.individual_runs]

    def show_all_results(self):
        all_results = torch.concat(
            [run.results for run in self.individual_runs], dim=-1
        )

        L, _ = all_results.shape

        max_result = all_results.max()

        big_img = [torch.ones((L, 1)) * (max_result / 2)]

        for run in self.individual_runs:
            big_img.append(run.results)
            big_img.append(torch.ones((L, 1)) * (max_result / 2))

        big_img = torch.cat(big_img, dim=-1)

        heads_str = ", ".join([f"L{layer}H{head}" for layer, head in self.target_nodes])

        imshow(
            big_img,
            title=f"Patch {self.target_location} on heads {heads_str}",
            # width=600,
            labels={"x": "Head", "y": "Layer"},
        )

    def show_single_dataset_results(self, i):
        self.individual_runs[i].show_results()
