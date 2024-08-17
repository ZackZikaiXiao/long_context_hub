import argparse
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--dataset_cache", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100)

    # model arguments
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_device_mapping", action="store_true")
    parser.add_argument("--max_generation_length", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--streaming_length", type=int, default=None)
    parser.add_argument("--truncation_side", type=str, default="right")
    parser.add_argument("--runtime_truncation", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--device_map", type=str, default=None)

    # dataset arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_group", type=str, default=None)
    parser.add_argument("--max_data_num", type=int, default=None)
    parser.add_argument("--start_data_from", type=int, default=None)
    parser.add_argument("--no_end_token", action="store_true")

    # training arguments
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_token_size", type=int, default=None)

    # evaluation arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--evaluate_metrics", action="store_true")
    parser.add_argument("--suppress_tokens", type=int, default=[], nargs="*")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--evaluate_positions", type=int, nargs="+")
    parser.add_argument("--structured_prompt", action="store_true")
    parser.add_argument("--api_token", type=str)

    # lambda attention arguments
    parser.add_argument("--use_lambda_attention", action="store_true")
    # parser.add_argument("--efficient_implementation", action="store_true")
    parser.add_argument("--local_branch", type=int, default=2048)
    parser.add_argument("--global_branch", type=int, default=100)
    parser.add_argument("--limit_distance", type=int, default=None)
    parser.add_argument("--triangle_offset", type=float, default=0.0)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--constant_answer", type=str, default="")
    parser.add_argument("--top_k_attention", type=int, default=None)
    parser.add_argument("--top_k_insert_at", type=int, default=2048)
    parser.add_argument("--top_k_from_layer", type=int, default=4)
    parser.add_argument("--top_k_to_layer", type=int, default=1000)

    args = parser.parse_args()

    set_seed(args.seed)
    return args
