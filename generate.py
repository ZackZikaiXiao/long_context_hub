from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import numpy as np
from torch import nn
from generate_replace import generate_replace
from transformers import AutoConfig


#Enable modifed generation function
generate_replace()   

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# 初始化模型和tokenizer
model_name = "./models/llama2-7B-4k"  # 替换为你的Llama模型路径
tokenizer = LlamaTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)
device = torch.device('cuda')
tokenizer.padding_side = "left"


model = LlamaForCausalLM.from_pretrained(model_name,
                                        config=config,
                                        torch_dtype=torch.bfloat16,   
                                        attn_implementation="flash_attention_2",
                                        device_map='auto')
special_tokens_dict = dict()
special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


def generate(model, prompts, temperature=1.0, top_p=0.9, top_k=50, max_length=100):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    attention_mask = inputs["attention_mask"]
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        # num_beams=5,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

prompts = [f"[INST]Who are you?[/INST]", f"[INST]What's color of an apple?[/INST]", f"[INST]How long is a pencil?[/INST]"]
# prompts = [f"Who are you?", f"What's color of an apple?", f"How long is the pencil?"]
generated_texts = generate(model, prompts, temperature=1.0, top_p=0.9, top_k=50, max_length=50)

print("\nGenerate:")
for i, text in enumerate(generated_texts):
    print(f"Prompt {i+1}: {text}")

