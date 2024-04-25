# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
from typing import List
import torch

sys.path.insert(1, "/root/successor_feature/")

# %%
from src.path_patching.multi_dataset_run import MultiDatasetRun
from src.plot_utils import imshow
from src.type_utils import AttnHead

# %%
N = 100
heads: List[AttnHead] = [(7, 11)]


# %%
values = MultiDatasetRun(N, heads, "v")


# %%
values.show_all_results()

# %%
values.corrupted_datasets[4].name

# %%
queries = MultiDatasetRun(N, heads, "q")

# %%
queries.show_all_results()


# %%
keys = MultiDatasetRun(N, heads, "k")

# %%
keys.show_all_results()

# %%
keys.show_single_dataset_results(6)

# %%
keys.show_single_dataset_results(0)


# %%
from src.activation import show_stats_for_string


# %%

show_stats_for_string("14. next 15. into 16. Hawaii")





# %%
queries.corrupted_datasets[2].name

# %%
[d.get_average_activation().item() for d in keys.corrupted_datasets]

# %%
