import itertools
import torch

from transformer_lens import utils
from typing import List
from functools import partial

from src.activation import get_linear_feature_activation_from_cache
from src.dataset.dataset import (
    Dataset,
    get_corrupted_normalizing_function_for_datasets,
    get_mended_normalizing_function_for_datasets,
)
from src.plot_utils import imshow
from src.type_utils import AttnHead
from src.model import model

from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from tqdm import tqdm
from enum import Enum

MLP_0 = utils.get_act_name("mlp_out", 0)


# Get activations that we treat as nodes in the computation graph
def node_filter(name: str) -> bool:
    # if name == "hook_embed":
    #     return True
    # elif name == MLP_0:
    #     return True

    return name.endswith("z")


def z_filter(name: str) -> bool:
    return name.endswith("z")


def edge_computation_hook(
    activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    isolated_source: AttnHead,
    base_dataset: Dataset,
    patch_dataset: Dataset,
    mean_patch=True,
):
    activation[...] = base_dataset.cache[hook.name][...]

    layer, head = isolated_source

    if hook.layer() == layer:
        if mean_patch:
            activation[:, :, head] = patch_dataset.get_mean_patch(
                str(hook.name), activation.shape[0]
            )[:, :, head]
        else:
            activation[:, :, head] = patch_dataset.cache[hook.name][:, :, head]

    return activation


def edge_patch_hook(
    activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    target_nodes: List[AttnHead],
    patch_cache: ActivationCache,
):
    heads_to_patch = [head for layer, head in target_nodes if layer == hook.layer()]

    # print("i'm being called!", heads_to_patch, hook.name)

    activation[:, :, heads_to_patch] = patch_cache[hook.name][:, :, heads_to_patch]

    # return torch.zeros_like(activation).to(activation.dtype)
    return activation


class PatchType(Enum):
    CORRUPT = 1
    MEND = 2


class SingleRun:
    _results = None

    def __init__(
        self,
        clean_dataset,
        corrupted_dataset,
        target_nodes: List[AttnHead],
        target_location: str,
        normalizing_function=None,
        patch_type=PatchType.CORRUPT,
    ):
        self.clean_dataset = clean_dataset
        self.corrupted_dataset = corrupted_dataset
        self.target_nodes = target_nodes
        self.patch_type = patch_type

        assert target_location in ("q", "k", "v", "z")

        self.target_location = target_location

        self.model = model

        if normalizing_function is not None:
            self.normalizing_function = normalizing_function
        else:
            if patch_type == PatchType.CORRUPT:
                self.normalizing_function = (
                    get_corrupted_normalizing_function_for_datasets(
                        self.clean_dataset, self.corrupted_dataset
                    )
                )
            else:
                self.normalizing_function = (
                    get_mended_normalizing_function_for_datasets(
                        self.clean_dataset, self.corrupted_dataset
                    )
                )

    def run(self):
        if self.patch_type == PatchType.CORRUPT:
            mean_patch = True

            base_dataset = self.clean_dataset
            patch_dataset = self.corrupted_dataset
        else:
            mean_patch = False

            base_dataset = self.corrupted_dataset
            patch_dataset = self.clean_dataset

        receiver_layers = [layer for layer, _ in self.target_nodes]
        receiver_hook_names = [
            utils.get_act_name(self.target_location, layer) for layer in receiver_layers
        ]
        receiver_hook_names_filter = lambda name: name in receiver_hook_names

        results = torch.zeros(min(receiver_layers), model.cfg.n_heads)

        for layer, head in tqdm(
            list(
                itertools.product(range(min(receiver_layers)), range(model.cfg.n_heads))
            )
        ):
            model.reset_hooks()

            edge_computation_hook_fn = partial(
                edge_computation_hook,
                isolated_source=(layer, head),
                base_dataset=base_dataset,
                patch_dataset=patch_dataset,
                mean_patch=mean_patch,
                # base_dataset=self.clean_dataset,
                # patch_dataset=self.corrupted_dataset,
            )

            model.add_hook(node_filter, edge_computation_hook_fn)

            _, edge_cache = model.run_with_cache(
                # self.clean_dataset.tokens, return_type=None
                base_dataset.tokens,
                return_type=None,
            )

            model.reset_hooks()

            edge_patch_hook_fn = partial(
                edge_patch_hook, target_nodes=self.target_nodes, patch_cache=edge_cache
            )

            model.add_hook(receiver_hook_names_filter, edge_patch_hook_fn)

            _, final_cache = model.run_with_cache(
                # self.clean_dataset.tokens, return_type=None
                base_dataset.tokens,
                return_type=None,
            )

            results[layer, head] = self.normalizing_function(
                get_linear_feature_activation_from_cache(final_cache)
            )

        model.reset_hooks()

        return results

    @property
    def results(self):
        if self._results is None:
            self._results = self.run()

        return self._results

    def show_results(self):
        heads_str = ", ".join(
            [f"L{layer}H{head}" for layer, head in self.target_nodes]
        )[:-2]

        imshow(
            self.results,
            title=f"Patch {self.target_location} on heads {heads_str}",
            width=600,
            labels={"x": "Head", "y": "Layer"},
        )
