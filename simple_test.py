import os
import sys
import json
import math
import torch
import argparse
import random
import numpy as np
from termcolor import colored
from transformers import AutoTokenizer
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--attn_type", type=str, default="RetroInfer",                                                      \
                        choices=["Full_Flash_Attn", "RetroInfer", "RetrievalAttention"], help="Attention method")
    parser.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",                  \
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",               \
                        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"], help="huggingface model name")
    parser.add_argument("--data_path", type=str, default="", help="Input json file path")
    parser.add_argument(
        "--token_budget_override",
        type=int,
        default=None,
        help="Fixed RetrievalAttention token budget. If unset, use ratio-derived budget.",
    )
    parser.add_argument(
        "--recall_only",
        action="store_true",
        help="Run prefill/index build only and report KNN parity recall summary.",
    )
    parser.add_argument(
        "--recall_input_tokens",
        type=int,
        default=8192,
        help="Input token length for recall-only mode.",
    )
    parser.add_argument(
        "--recall_min_recall",
        type=float,
        default=None,
        help="Optional minimum weighted recall threshold for recall-only mode.",
    )
    args = parser.parse_args()
    
    return args


def build_recall_only_inputs(tokenizer, batch_size: int, token_len: int):
    token_len = max(1, int(token_len))
    vocab_size = int(len(tokenizer))
    if vocab_size <= 8:
        raise RuntimeError(f"Unexpectedly small tokenizer vocab_size={vocab_size}")

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = 1
    bos_id = int(max(0, min(vocab_size - 1, int(bos_id))))

    low = 8 if vocab_size > 16 else 1
    span = max(1, vocab_size - low)
    base = (torch.arange(token_len, dtype=torch.long) % span) + low
    base[0] = bos_id
    input_ids = base.unsqueeze(0).repeat(int(batch_size), 1)
    attention_mask = torch.ones((int(batch_size), token_len), dtype=torch.long)
    return input_ids, attention_mask


def load_model(model_name, max_len, dtype, device):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm


def generate_config(model_name, context_len, attn_type, token_budget_override=None):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)

    # Upstream configs may not include RetrievalAttention entries.
    if attn_type == "RetrievalAttention" and attn_type not in original_config:
        default_static_start = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_START", "128"))
        default_static_end = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_END", "512"))
        default_q_knn = int(os.environ.get("RETRIEVALATTN_Q_KNN", "8"))
        default_key_degree = int(os.environ.get("RETRIEVALATTN_KEY_DEGREE", "8"))
        original_config[attn_type] = {
            "static_pattern_start": max(0, default_static_start),
            "static_pattern_end": max(0, default_static_end),
            "q_knn": max(1, default_q_knn),
            "key_degree": max(1, default_key_degree),
            "token_budget": 1,
        }
    
    n_clusters = max(int(context_len/16), 1)
    n_segments = max(int(context_len/8192), 1)
    # compute the nearest multiple of (n_segments*32)
    lower = (n_clusters // (n_segments*32)) * (n_segments*32)
    upper = lower + (n_segments*32)
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    nprobe = max(int(n_clusters*0.018), 1)

    if attn_type == 'RetroInfer':
        original_config[attn_type]['n_centroids'] = n_clusters
        original_config[attn_type]['n_segment'] = n_segments
        original_config[attn_type]['nprobe'] = nprobe
        original_config[attn_type]['cache_cluster_num'] = nprobe*3
        original_config[attn_type]['max_compute_cluster_num'] = max(int(n_clusters/4), nprobe)
    if attn_type == 'RetrievalAttention':
        if token_budget_override is not None and token_budget_override > 0:
            token_budget = int(token_budget_override)
        else:
            avg_cluster_size = max(int((context_len - (original_config[attn_type]["static_pattern_start"] + original_config[attn_type]["static_pattern_end"])) / n_clusters), 1)
            token_budget = max(int(nprobe * avg_cluster_size), 1)
        original_config[attn_type]['token_budget'] = token_budget
        # keep q_knn/key_degree from config if present
    
    if attn_type != "Full_Flash_Attn":
        print(original_config[attn_type])
    
    return original_config


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)

    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
    device = args.device
    data_path = args.data_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.recall_only:
        input_ids, attention_masks = build_recall_only_inputs(
            tokenizer=tokenizer,
            batch_size=batch_size,
            token_len=args.recall_input_tokens,
        )
        groundtruths = []
        gen_len = 1
        input_len = input_ids.shape[1]
        max_len = input_len + gen_len
        print(colored(f"Recall-only mode: input_tokens={input_len}, gen_len={gen_len}", "yellow"))
        attn_config = generate_config(
            model_name,
            input_len,
            attn_type,
            token_budget_override=args.token_budget_override,
        )
    else:
        # load input data
        if data_path == "":
            TEST_FILE = os.path.join(PROJECT_ROOT, "simple_test_data.json")
        else:
            TEST_FILE = os.path.join(PROJECT_ROOT, f"{data_path}")
        print(colored(f"Loading test data from {TEST_FILE}", 'yellow'))
        data = json.load(open(TEST_FILE))   # [{"input": str, "outputs": str}, ...]
        prompt = []
        groundtruth = []
        for dd in data:
            prompt.append(dd['input'])
            groundtruth.append(dd['outputs'])
        
        # copy to fit batch size
        copy_round = math.ceil(batch_size/len(prompt))
        prompts = []
        groundtruths = []
        for _ in range(copy_round):
            prompts.extend(prompt)
            groundtruths.extend(groundtruth)
        prompts = prompts[:batch_size]
        groundtruths = groundtruths[:batch_size]

        # tokenize input data
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_masks = inputs.attention_mask

        input_len = input_ids.shape[1]
        gen_len = args.gen_len
        max_len = input_len + gen_len
        print(colored(f"Input length: {input_len}", 'yellow'))

        if data_path == "":
            attn_config = generate_config(
                model_name,
                122880,
                attn_type,
                token_budget_override=args.token_budget_override,
            )
        else:
            attn_config = generate_config(
                model_name,
                input_len,
                attn_type,
                token_budget_override=args.token_budget_override,
            )

    llm = load_model(model_name, max_len, dtype, device)
    out = llm.generate(attention_type=attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=gen_len, attn_config=attn_config)

    if args.recall_only:
        parity_summary = None
        if hasattr(llm, "kv_cache") and hasattr(llm.kv_cache, "get_parity_summary"):
            parity_summary = llm.kv_cache.get_parity_summary(reset=False)
        print("[RECALL] parity_summary_json=" + json.dumps(parity_summary, sort_keys=True))
        if args.recall_min_recall is not None:
            if parity_summary is None:
                raise RuntimeError("recall_min_recall set but parity summary is unavailable.")
            recall_weighted = parity_summary.get("recall_weighted", None)
            if recall_weighted is None:
                raise RuntimeError(
                    "recall_min_recall set but parity summary has no recall_weighted value."
                )
            if float(recall_weighted) < float(args.recall_min_recall):
                raise RuntimeError(
                    f"recall_weighted={float(recall_weighted):.6f} below threshold "
                    f"{float(args.recall_min_recall):.6f}"
                )
    else:
        result = tokenizer.batch_decode(out, skip_special_tokens=True)
        for gt, res in zip(groundtruths, result):
            print(colored(f"Answer: {gt}", 'yellow'))
            print(f"{[res]}")
    
