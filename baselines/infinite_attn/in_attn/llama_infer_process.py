
# Written by Yukang Chen
# Core code based on https://github.com/CStanKonrad/long_llama
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

import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
from peft import PeftModel
from llama_attn_replace import replace_llama_attn
from transformers import LlamaTokenizer

from attention.modeling_llama_infinite_in_attn import LlamaForCausalLM
# from transformers.models.llama.modeling_llama import LlamaForCausalLM



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/home/zikaixiao/zikaixiao/LongLoRA-main/memonet_output")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=8192*2, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=True, help='whether to use flash attention 2')
    parser.add_argument('--max_tokens', type=int, default=8192*2, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=3, help='number of repeat testing for each length')


    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval(model, tokenizer, device, use_cache, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=8192).input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS

    # out = model(answer_ids)
    # for token_idx in range(answer_ids["input_ids"].shape[1]):
    #     print("----")
    #     print("Input:", tokenizer.batch_decode(answer_ids["input_ids"][:, :token_idx + 1]))
    #     print("Prediction:", tokenizer.batch_decode(torch.tensor([[out.logits.argmax(-1)]])))
    outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False)
    generation_output = model.generate(
        input_ids=input_ids, labels=input_ids, max_new_tokens=30, num_beams=1, use_cache=False
    )
    model_answer = generation_output[0, -30:].cpu()

    correct_answer = tokenizer.decode(answer_ids[0].cpu())
    model_answer = tokenizer.decode(model_answer.cpu())

    is_correct = correct_answer in model_answer
    # print(f"The correct answer is: {correct_answer}")
    # print(f"The model answer is: {model_answer}, is_correct : {is_correct}")
    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)
    
    config = transformers.AutoConfig.from_pretrained(
    args.base_model,
    cache_dir=args.cache_dir,
    )

    model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        padding="longest",
        max_length=args.context_size,
    )

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        passed_tests = 0
        total_tokens = 0
        if n_garbage < (int(3.75 * (total_test_points + 1) * args.interval // 1024 * 1024)) / 2:
            continue
        for i in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval(model, tokenizer, device, use_cache=not args.flash_attn, n_garbage=n_garbage, seed=i)
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens//args.num_tests
        accuracy = float(passed_tests)/args.num_tests
        print("accuracy on the token length %d is %f"%(avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)


if __name__ == "__main__":
    args = parse_config()
    main(args)
