from typing import Optional
import torch
import datasets
import transformers
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler, WeightedRandomSampler
import numpy as np

class LengthSampler(Sampler):
    def __init__(self, data_source, lengths, weights):
        self.data_source = data_source
        self.lengths = lengths
        self.weights = weights
        self.sorted_indices = np.argsort(self.lengths)
        self.current_index = 0

    def __iter__(self):
        sampled_indices = []
        total_samples = len(self.sorted_indices)
        batch_size = len(self.data_source) // 10  # 分成10个阶段逐步增加样本长度
        while self.current_index < total_samples:
            end_index = min(self.current_index + batch_size, total_samples)
            current_indices = self.sorted_indices[self.current_index:end_index]
            sampled_indices.extend(current_indices)
            self.current_index = end_index
            yield from sampled_indices

    def __len__(self):
        return len(self.data_source)

def create_length_sampler(data_source, lengths):
    # 将权重均匀设置为1，保证均衡采样
    weights = np.ones_like(lengths, dtype=np.float32)
    return LengthSampler(data_source, lengths, weights)

def replace_upsampling_with_length_sampler(lengths):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                dataset_lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                dataset_lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=dataset_lengths,
                model_input_name=model_input_name,
            )
        else:
            return create_length_sampler(self.train_dataset, lengths)
    transformers.Trainer._get_train_sampler = _get_train_sampler
