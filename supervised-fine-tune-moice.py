# Written by Zikai Xiao
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

## LoRaMEL (LoRA Multi-Experts for Long contexts)
import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from torch.distributed import barrier
import random
from attention.modeling_llama_moice import LlamaForCausalLM


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f: str = "all", mode="r"):
    random.seed(42)
    with open(f, mode) as file:
        jdict = json.load(file)

    num_samples = int(len(jdict) * 0.5)
    # num_samples = 64

    if len(jdict) < num_samples:
        raise ValueError(f"The dataset contains only {len(jdict)} records, but {num_samples} samples were requested.")
    
    return random.sample(jdict, num_samples) 



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
    model_name_or_path: Optional[str] = field(default="./tinyllama_weights/tinyllama-1.1B-16k-ft")
    topk : int = field(default=7)
    expert_nums : int = field(default=7)
    base_set: str = field(default="[10000,17500,18000,19000,20000,22500,25000]")    # 57.22571839416256
    # base_set: str = field(default="[10000, 48333, 86667, 125000, 163333, 201667, 240000]")    
    router_aux_loss_coef : float =field(default=0.3)

@dataclass
class DataArguments:
    data_path: str = field(default="./dataset/LongAlpaca-12k.json", metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    source_model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_max_length: int = field(
        default=8192*2,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    only_train_gate: bool = True,

    use_databricks: bool = field(
        default=False,        
        metadata={"help": "Whether databricks for training."},
    )
    output_dir: str = field(default="./tinyllama-1.1B-16k-ft-sft-moice")
    num_train_epochs: int = field(default=5)        # 5
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=98) # 98
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=1e-4)  # official longlora: 2e-5; tiny llama:4e-4
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.2)
    # warmup_steps: int = field(default=20)
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

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        list_data_dict = jload(data_path)

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
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.use_databricks:
        torch.cuda.set_device(int(os.environ["RANK"]))
        import mlflow
        experiment = mlflow.set_experiment(os.environ["EXPERIMENT_PATH"])


    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path,cache_dir=training_args.cache_dir,trust_remote_code=True)
    config._flash_attn_2_enabled=True
    config.topk = model_args.topk
    config.output_router_logits = True
    config.router_aux_loss_coef = model_args.router_aux_loss_coef
    config.base_set = json.loads(model_args.base_set)
    config.expert_nums = model_args.expert_nums

    # orig_rope_scaling = getattr(config, "rope_scaling", None)
    # if orig_rope_scaling is None:
    #     orig_rope_scaling = {"factor": 1}
    # orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    # orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # if orig_ctx_len:
    #     orig_ctx_len *= orig_rope_scaling_factor
    #     if training_args.model_max_length > orig_ctx_len:
    #         scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
    # scaling_factor

    # self.base * (
    #             (scaling_factor * 8) - (scaling_factor - 1)
    #         ) ** (2048 / (2048 - 2))
    
    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     torch_dtype=torch.bfloat16,     # torch.float32, bfloat16
    #     use_flash_attention_2=training_args.use_flash_attention_2
    #     # device_map='cpu'               ########################### 并行是没有的
    # )
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,     # torch.float32, bfloat16
        attn_implementation="flash_attention_2",
        # use_flash_attention_2=training_args.use_flash_attention_2
        # device_map='auto'               ########################### 并行是没有的
    )

    if training_args.only_train_gate:
        for n,p in model.named_parameters():
            p.requires_grad_(False)
            
        for n,p in model.named_parameters():
            if "self_attn.gate" in n:
                p.requires_grad_(True)
                print(n)
        model.enable_input_require_grads() 


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

    if training_args.use_databricks:
        rank = int(os.environ.get('RANK', -1))
        if rank > 0:
            barrier()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.use_databricks:
        if rank == 0:
            barrier()
            
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer.add_callback(SavePeftModelCallback)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()

