# import random


import random
import copy

from datasets import Dataset, DatasetDict
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset


def poison_partition(partition: P2PFLDataset, poison_fraction: float = 1.0, trigger_src: int = 7, target: int = 5) -> P2PFLDataset:
    """
    Given a P2PFLDataset partition, return a new partition where in the train split each sample
    with label equal to trigger_src is relabeled as target with probability poison_fraction.
    
    This function uses partition.get_num_samples(train=True) to determine the number of samples.
    """
    # Get the number of training samples.
    n = partition.get_num_samples(train=True)
    poisoned_train_items = []
    
    # Iterate over indices using get_num_samples.
    for i in range(n):
        # Use the partition's getter (which does not require __len__).
        item = partition.get(i, train=True).copy()
        if "label" in item and item["label"] == trigger_src and random.random() < poison_fraction:
            print(f"Changing label {item['label']} to {target} at index {i}")
            item["label"] = target
        poisoned_train_items.append(item)
    
    # Create a new Hugging Face Dataset from the poisoned items.
    poisoned_train_ds = Dataset.from_list(poisoned_train_items)
    
    # Retrieve (and reuse) the original test split.
    test_ds = partition._data.get(partition._test_split_name)
    
    # Build a new DatasetDict preserving the original split names.
    new_data = DatasetDict({
        partition._train_split_name: poisoned_train_ds,
        partition._test_split_name: test_ds
    })
    
    # Return a new P2PFLDataset using the new data.
    return P2PFLDataset(new_data, train_split_name=partition._train_split_name, test_split_name=partition._test_split_name)


# class MnistBackdoor:
#     """
#     A wrapper for a Hugging Face MNIST dataset that relabels a fraction of samples
#     originally labeled as 7 to instead be labeled as 5.

#     Args:
#         base_dataset: The original Hugging Face dataset (e.g., loaded with load_dataset).
#         poison_fraction (float): Fraction of samples with label 7 to be relabelled as 5 (0.0 to 1.0).

#     Example:
#         from datasets import load_dataset
#         base_ds = load_dataset("p2pfl/MNIST", split="train")
#         backdoored_ds = MnistBackdoor(base_ds, poison_fraction=0.5)
#         sample = backdoored_ds[0]
#     """
#     def __init__(self, base_dataset, poison_fraction=1.0):
#         self.base = base_dataset
#         self.poison_fraction = poison_fraction

#     def __len__(self):
#         return len(self.base)

#     def __getitem__(self, index):
#         # Get the original item
#         item = self.base[index].copy()  # copy to avoid modifying the original
#         # Check if the label is 7; if so, possibly change to 5
#         if "label" in item and item["label"] == 7:
#             if random.random() < self.poison_fraction:
#                 item["label"] = 5  # relabel 7 as 5
#         return item