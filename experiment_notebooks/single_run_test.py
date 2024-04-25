# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
import torch

sys.path.insert(1, "/root/successor_feature/")


# %%
from src.path_patching.single_run import SingleRun

# %%
from src.dataset.clean_dataset import CleanDataset
from src.dataset.name_corrupted_datasets import LastNameCorruptedDataset, PrefixNamesCorruptedDataset, SecondNameCorruptedDataset, FirstNameCorruptedDataset
from src.dataset.number_corrupted_dataset import LastNumberCorruptedDataset, SecondNumberCorruptedDataset, FirstNumberCorruptedDataset

# %%
SecondNumberCorruptedDataset(100).get_average_activation()


# %%
from src.plot_utils import imshow

# %%
CleanDataset(100).get_average_activation()



# %%
from src.activation import show_stats_for_string


# %%
N = 100

clean_dataset = CleanDataset(N)
corrupted_dataset = FirstNameCorruptedDataset(N)
# corrupted_dataset = SecondNumberCorruptedDataset(N)
# corrupted_dataset = LastNumberCorruptedDataset(N)
# corrupted_dataset = FirstNumberCorruptedDataset(N)

# corrupted_dataset = PrefixNamesCorruptedDataset(N)

# %%
show_stats_for_string(corrupted_dataset.prompt_strings[4])

# %%
corrupted_dataset.get_average_activation(), clean_dataset.get_average_activation()



# %%
run = SingleRun(clean_dataset, corrupted_dataset, [(7, 11)], "k")
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
run.results.shape

# %%
L, H = run.results.shape


# %%
mm = torch.ones(L, 1) * run.results.max()

dd = torch.concat([
        run.results, 
        mm,
        run.results, 
        mm,
        run.results, 
        mm,
        run.results, 
        mm,
        run.results, 
        mm,
        run.results, 
        mm,
    ],
    dim=-1
)


# %%
imshow(dd)

# %%
dd.shape

# %%
