from typing import Optional
import torch
import datasets
import transformers
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler
import numpy as np


def create_custom_get_train_sampler(sample_weights):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            print("Enter my sampler")
            return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print("finish replacement")
    return _get_train_sampler

def replace_upsampling_with_random_sampling(sample_weights):
    print('use upsampling')
    
    # Create a custom get_train_sampler method with the sample_weights passed in
    new_get_train_sampler = create_custom_get_train_sampler(sample_weights)
    # Replace the existing _get_train_sampler method with the new one
    transformers.Trainer._get_train_sampler = new_get_train_sampler