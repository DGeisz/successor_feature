# %%
%load_ext autoreload
%autoreload 2

# %%
import sys

sys.path.insert(1, "/root/successor_feature/")


# %%
from src.path_patching.single_run import SingleRun

# %%
from src.dataset.clean_dataset import CleanDataset
from src.dataset.name_corrupted_datasets import LastNameCorruptedDataset, PrefixNamesCorruptedDataset
from src.dataset.number_corrupted_dataset import LastNumberCorruptedDataset, SecondNumberCorruptedDataset, FirstNumberCorruptedDataset

# %%
from src.activation import show_stats_for_string


# %%
N = 200

clean_dataset = CleanDataset(N)
# corrupted_dataset = LastNameCorruptedDataset(N)
# corrupted_dataset = SecondNumberCorruptedDataset(N)
# corrupted_dataset = LastNumberCorruptedDataset(N)
corrupted_dataset = FirstNumberCorruptedDataset(N)

# corrupted_dataset = PrefixNamesCorruptedDataset(N)

# %%
show_stats_for_string(corrupted_dataset.prompt_strings[4])

# %%
corrupted_dataset.get_average_activation(), clean_dataset.get_average_activation()



# %%
run = SingleRun(clean_dataset, corrupted_dataset, [(7, 11)], "v")
run.show_results()

# %%
clean_dataset.get_average_activation().item(), corrupted_dataset.get_average_activation().item()


# %%

# %%
_, cache = run.model.run_with_cache(run.clean_dataset.tokens)

# %%
cache['z', 0][:, :, [1]].shape



# %%
clean_dataset.tokens.shape

# %%
run.model(clean_dataset.tokens)

# %%
