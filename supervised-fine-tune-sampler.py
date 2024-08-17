# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import matplotlib.pyplot as plt

import torch
import transformers
from torch.utils.data import Dataset
# from transformers import Trainer, DataCollatorForLanguageModeling
from trainer import Trainer
from torch.distributed import barrier
import random
from transformers import PreTrainedTokenizer
import numpy as np
from resampling_replace import replace_upsampling_with_random_sampling
from llama_flash_attn import replace_llama_attn_with_flash_attn
from torch.utils.data import IterableDataset
from datasets import load_dataset

# from save_callback import SavePeftModelCallback

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

weigh_mode = ["length_upsampling", "cherry_weighting", "lidd_weighting", "prolong_weighting", 
              "lidd_upsampling", ][0]
partial_training_enable = False
replace_upsampling_enable = False


def load_json(data_path: str, tokenizer: PreTrainedTokenizer) -> list:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_samples = int(len(data) * 1)
    sampled_indices = random.sample(range(len(data)), num_samples)
    sampled_data = [data[i] for i in sampled_indices]
    return sampled_data

def load_and_resample_dataset(data_path: str, tokenizer: PreTrainedTokenizer) -> list:
    """Load dataset, perform square-root sampling based on binned sample lengths, and save the distribution plot."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    part = 5
    segment_num = 10
    random.seed(42)  # Set a fixed seed for reproducibility
    random.shuffle(data)
    part_size = len(data) // segment_num
    data_parts = [data[i * part_size:(i + 1) * part_size] for i in range(segment_num)]
    leftover_samples = data[segment_num * part_size:]
    for i, sample in enumerate(leftover_samples):
        data_parts[0].append(sample)
    sampled_data = data_parts[part]
    data = sampled_data

    # num_samples = int(len(data) * 0.1)
    # sampled_indices = random.sample(range(len(data)), num_samples)
    # sampled_data = [data[i] for i in sampled_indices]
    # data = sampled_data
    if weigh_mode == "length_upsampling":
        sample_lengths = np.array([len(sample['instruction'].replace(' ', '') + sample['output'].replace(' ', '')) for sample in data])
        # 将样本长度进行分段（分箱）
        num_bins = 100
        min_length = sample_lengths.min()
        max_length = sample_lengths.max()
        bins = np.linspace(min_length, max_length, num_bins + 1)
        binned_lengths = np.digitize(sample_lengths, bins) - 1  # 得到每个样本所属的bin索引
        # 统计每个bin的频率
        bin_freq = np.bincount(binned_lengths, minlength=num_bins)
        # 防止 bin_freq 中包含0
        bin_freq = np.where(bin_freq == 0, 1e-6, bin_freq)
        # 计算每个bin频率的平方根
        sqrt_freq = np.power(bin_freq, 0.25) / bin_freq
        # 归一化平方根频率
        total_sqrt_freq = np.sum(sqrt_freq)
        normalized_probabilities = sqrt_freq / total_sqrt_freq
        # 分配每个样本的权重
        sample_weights = normalized_probabilities[binned_lengths]
        # np.save("./upsampling_weights.npy", sample_weights)
        return data, sample_weights
    elif weigh_mode == "cherry_weighting":
        lds_array = np.load("./Cherry_LLM/cherry_array.npy")
        sampled_lds_array = np.array([lds_array[i] for i in sampled_indices])
        plt.figure(figsize=(10, 6))
        plt.hist(lds_array, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title('Distribution of LDS Array')
        plt.xlabel('LDS Value')
        plt.ylabel('Frequency')
        plt.savefig('./cherry_distribution.png')
        return data, sampled_lds_array
    
    elif weigh_mode == "lidd_weighting":
        lds_array = np.load("./data_augument/lidd/results-llama/lidd_array.npy")
        sampled_array = np.array([lds_array[i] for i in sampled_indices])
        return data, sampled_array
    elif weigh_mode == "prolong_weighting":
        lds_array = np.load("./data_augument/prolong/analysis/lds_array.npy")
        sampled_lds_array = np.array([lds_array[i] for i in sampled_indices])
        # 都加上500，避免为0的情况
        sampled_lds_array += 500
        plt.figure(figsize=(10, 6))
        plt.hist(lds_array, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title('Distribution of LDS Array')
        plt.xlabel('LDS Value')
        plt.ylabel('Frequency')
        plt.savefig('./prolong_distribution.png')
        return data, np.power(sampled_lds_array, 0.5)
    elif weigh_mode == "lidd_upsampling":
        length_prime = np.load("./data_augument/lidd/results-llama/lidd_array.npy")
        length_prime = np.array([length_prime[i] for i in sampled_indices])
        sample_lengths = length_prime
        sample_lengths = sample_lengths * np.array([len(sample['instruction'].replace(' ', '') + sample['output'].replace(' ', '')) for sample in data])
        # 计算每个样本的长度
        # sample_lengths = np.array([len(sample['instruction'].replace(' ', '') + sample['output'].replace(' ', '')) for sample in data])
        
        # 将样本长度进行分段（分箱）
        num_bins = 100
        min_length = sample_lengths.min()
        max_length = sample_lengths.max()
        bins = np.linspace(min_length, max_length, num_bins + 1)
        binned_lengths = np.digitize(sample_lengths, bins) - 1  # 得到每个样本所属的bin索引
        # 统计每个bin的频率
        bin_freq = np.bincount(binned_lengths, minlength=num_bins)
        # 防止 bin_freq 中包含0
        bin_freq = np.where(bin_freq == 0, 1e-6, bin_freq)
        # 计算每个bin频率的平方根
        sqrt_freq = np.power(bin_freq, 0.25) / bin_freq
        # 归一化平方根频率
        total_sqrt_freq = np.sum(sqrt_freq)
        normalized_probabilities = sqrt_freq / total_sqrt_freq
        # 分配每个样本的权重
        sample_weights = normalized_probabilities[binned_lengths]
        # np.save("./upsampling_weights.npy", sample_weights)
        return data, sample_weights


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./tinyllama_weights/tinyllama-1.1B-16k-ft-qkvo")
    # /home/zikaixiao/zikaixiao/llmmemo/Llama2-7b-chat
    # /home/zikaixiao/zikaixiao/llmmemo/TinyLlama-1.1B-Chat-v1.0
    # Llama2-7b-chat
    # TinyLlama-1.1B-Chat-v1.0


@dataclass
class DataArguments:
    data_path: str = field(default="./data_augument/aeda_LongAlpaca-12k.json", metadata={"help": "Path to the training data."})
    filter_mode: str = field(default="all", metadata={"help": "Filtering mode for the dataset. Options: all, 0-4k, 4k-10k, 10k-18k, not-0-4k, not-4k-10k, not-10k-plus"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 2,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attention_2: bool = field(
        default=True,           # ##############################
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=False,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="mlp",   # embed,norm,q_proj,k_proj,v_proj,o_proj,mlp
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    output_dir: str = field(default="./TinyLlama-1.1B-16k-test")
    num_train_epochs: int = field(default=1)        # 5    100条数据时, Epoch为15，1000条数据时, Epoch为10，10000条数据时, Epoch为2。
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10) # 98
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=2e-5)  # official longlora: 2e-5; tiny llama:4e-4
    weight_decay: float = field(default=0.0)
    # warmup_steps: int = field(default=20)
    warmup_ratio: float = field(default=0.1)    # epoch * 1%
    lr_scheduler_type: str = field(default="constant_with_warmup")
    logging_steps: int = field(default=1)
    deepspeed: str = field(default="ds_configs/stage2.json")      ####################### 并行是有的
    bf16: bool = field(default=True)  # 注意，`bf16` 和 `tf32` 参数需要根据 transformers 的版本和支持进行调整
    # tf32: bool = field(default=True)
    # max_steps: int = field(default=1000)
    

    

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, filter_mode: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        list_data_dict, sample_weights = load_and_resample_dataset(data_path, tokenizer)
        
        if replace_upsampling_enable:
            replace_upsampling_with_random_sampling(sample_weights)

        logging.warning("Formatting inputs...")

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_llama2"], PROMPT_DICT["prompt_llama2"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class SupervisedIterableDataset(IterableDataset):
    """Iterable Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, filter_mode: str):
        super(SupervisedIterableDataset, self).__init__()
        logging.warning("Loading data...")

        self.data_path = data_path
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self):
        list_data_dict = load_json(self.data_path, self.tokenizer)

        logging.warning("Formatting inputs...")

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_llama2"], PROMPT_DICT["prompt_llama2"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        self.data_dict = preprocess(sources, targets, self.tokenizer)

    def __iter__(self):
        try:
            for i in range(len(self.data_dict["input_ids"])):
                yield dict(input_ids=self.data_dict["input_ids"][i], labels=self.data_dict["labels"][i])
        except Exception as e:
            logging.error(f"An error occurred during iteration: {e}")

    def __len__(self):
        return len(self.data_dict["input_ids"])
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, filter_mode=data_args.filter_mode)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # replace_llama_attn_with_flash_attn()  
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,     # torch.float32, bfloat16
        attn_implementation="flash_attention_2"
        # device_map='cpu'               ########################### 并行是没有的
    )
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(DEVICE)

    # from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    # # Estimate memory needs for the model across the specified hardware setup
    # estimate_zero3_model_states_mem_needs_all_live(
    #     model=model,
    #     num_gpus_per_node=8,
    #     num_nodes=1
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if partial_training_enable:
        def set_requires_grad(model, requires_grad=False):
            for param in model.parameters():
                param.requires_grad_(requires_grad)
        set_requires_grad(model, requires_grad=False)
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer.add_callback(SavePeftModelCallback)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
