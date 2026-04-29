import argparse
import collections
import inspect
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_CACHE_DIR / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE_DIR / "transformers"))
for _cache_dir in (
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
):
    Path(_cache_dir).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


CODEBOOK = [
    "red", "blue", "green", "gold", "silver", "black", "white", "orange",
    "purple", "yellow", "brown", "pink", "cyan", "lime", "coral", "navy",
]
CODEBOOK_SET = set(CODEBOOK)
ENTRY_RE = re.compile(r"^ENTRY\s+(\d+):\s+([a-z]+)\s*$", re.MULTILINE)
ANSWER_RE = re.compile(r"^ANSWER\s+(\d+):\s+([a-z]+)\s*$", re.MULTILINE)


_ROAR_BACKEND_MODULE = None


def load_roar_backend_module():
    """Load RoarGraph wrappers without importing cache_hub.__init__ on login nodes."""
    global _ROAR_BACKEND_MODULE
    if _ROAR_BACKEND_MODULE is not None:
        return _ROAR_BACKEND_MODULE
    module_path = Path(__file__).resolve().parents[1] / "cache_hub" / "roargraph_cpp_backend.py"
    spec = importlib.util.spec_from_file_location("_hf_roargraph_cpp_backend", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RoarGraph backend wrapper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_hf_roargraph_cpp_backend", module)
    spec.loader.exec_module(module)
    _ROAR_BACKEND_MODULE = module
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="HF-native generated-memory benchmark with model inventory and hook tracing."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument(
        "--hf_language_model_only",
        action="store_true",
        help="Prefer text/causal-LM classes before vision-language conditional-generation classes when available.",
    )
    parser.add_argument("--generation_mode", type=str, default="manual_cache", choices=["manual_cache", "generate_reprefill"])
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_entries", type=int, default=24)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--prefill_filler_repeats", type=int, default=0)
    parser.add_argument("--min_prompt_tokens", type=int, default=0)
    parser.add_argument("--generation_margin_tokens", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="generated_memory_hf_eval_result")
    parser.add_argument("--inventory_only", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    parser.add_argument("--tokenizer_only", action="store_true")
    parser.add_argument("--trace_attention", action="store_true")
    parser.add_argument("--trace_prefill", action="store_true")
    parser.add_argument("--trace_decode_steps", type=int, default=8)
    parser.add_argument("--trace_max_records", type=int, default=256)
    parser.add_argument(
        "--hf_attention_mode",
        type=str,
        default="native",
        choices=["native", "oracle_topk", "graph_topk", "graph_topk_roar"],
        help=(
            "HF attention backend. oracle_topk uses exact dynamic top-k; graph_topk "
            "uses a Python prefill K-K graph plus online birth-time edges; graph_topk_roar "
            "uses RoarGraph CSR build/search wrappers. Sparse modes replace decode "
            "attention in replaceable full-attention modules only."
        ),
    )
    parser.add_argument("--hf_sparse_topk", type=int, default=128)
    parser.add_argument("--hf_sparse_static_prefix", type=int, default=128)
    parser.add_argument("--hf_sparse_static_suffix", type=int, default=512)
    parser.add_argument("--hf_graph_degree", type=int, default=16)
    parser.add_argument("--hf_graph_visit_budget", type=int, default=256)
    parser.add_argument("--hf_graph_seed_count", type=int, default=32)
    parser.add_argument("--hf_graph_online_edges", type=int, default=16)
    parser.add_argument(
        "--hf_graph_search_backend",
        type=str,
        default="cuda_group",
        choices=["cpp", "cuda_group", "cuda_fullgpu"],
        help=(
            "Search backend for graph_topk_roar. cuda_group uses native CUDA scoring "
            "with CPU frontier bookkeeping; cuda_fullgpu uses the RoarGraph full-GPU traversal kernel."
        ),
    )
    parser.add_argument("--hf_graph_candidate_target", type=int, default=0)
    parser.add_argument("--hf_graph_expand_width", type=int, default=32)
    parser.add_argument("--hf_graph_min_visits", type=int, default=32)
    parser.add_argument("--hf_graph_frontier_topn", type=int, default=128)
    parser.add_argument("--hf_graph_stop_patience", type=int, default=1)
    parser.add_argument("--hf_graph_stop_margin", type=float, default=0.0)
    parser.add_argument("--hf_graph_roar_cand_limit", type=int, default=32)
    parser.add_argument("--hf_graph_roar_enhance_limit", type=int, default=32)
    parser.add_argument("--hf_graph_roar_entry", type=str, default="hub", choices=["hub", "first"])
    parser.add_argument("--hf_graph_roar_threads", type=int, default=0)
    parser.add_argument(
        "--answer_prefix_scaffold",
        action="store_true",
        help="Inject answer prefixes one by one during answer decoding, matching generated_memory_eval.py.",
    )
    parser.add_argument(
        "--answer_constrained_codebook",
        action="store_true",
        help="With answer-prefix scaffold, choose each answer from the color codebook by constrained next-token selection.",
    )
    parser.add_argument(
        "--force_max_decode_steps",
        action="store_true",
        help="Latency-only mode: ignore early stop markers and run the full ledger/answer decode budgets.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def ensure_psutil_virtual_memory():
    try:
        import psutil
    except Exception:
        return
    if hasattr(psutil, "virtual_memory"):
        return

    fields = "total available percent used free active inactive buffers cached shared slab"
    svmem = collections.namedtuple("svmem", fields)

    def _sysconf_bytes(name, default=0):
        try:
            value = os.sysconf(name)
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    def virtual_memory():
        page_size = _sysconf_bytes("SC_PAGE_SIZE", 4096)
        phys_pages = _sysconf_bytes("SC_PHYS_PAGES", 0)
        avail_pages = _sysconf_bytes("SC_AVPHYS_PAGES", phys_pages)
        total = int(max(0, phys_pages) * max(1, page_size))
        available = int(max(0, avail_pages) * max(1, page_size))
        used = max(0, total - available)
        percent = (100.0 * float(used) / float(total)) if total > 0 else 0.0
        return svmem(
            total,
            available,
            percent,
            used,
            available,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    psutil.virtual_memory = virtual_memory


def build_ledger_prompt(sample_idx: int, num_entries: int, filler_repeats: int):
    filler = ""
    if filler_repeats > 0:
        filler_line = (
            "FILLER BLOCK. Ignore this block. It exists only to make the prompt longer. "
            "Do not copy it into the answer.\n"
        )
        filler = filler_line * int(filler_repeats)
    codebook_text = ", ".join(CODEBOOK)
    return (
        "You must follow the format exactly.\n"
        f"{filler}"
        f"Write exactly {int(num_entries)} ledger lines.\n"
        "Each ledger line must have exactly this format:\n"
        "ENTRY i: red\n"
        "Value rules:\n"
        f"- the value must be exactly one word from this list: {codebook_text}\n"
        "- repeats are allowed\n"
        "- do not add punctuation or explanations\n"
        "- do not skip or renumber entries\n"
        f"After ENTRY {int(num_entries)}, write exactly this line:\n"
        "END LEDGER\n"
        "Stop immediately after END LEDGER.\n"
        "Begin with ENTRY 1 now.\n"
        f"# SAMPLE_ID={sample_idx}\n"
    )


def build_question_prompt(query_positions, answer_prefix_scaffold: bool = False):
    question_lines = [
        f"QUESTION {i + 1}: What was the value in ENTRY {int(pos)}?"
        for i, pos in enumerate(query_positions)
    ]
    if answer_prefix_scaffold:
        codebook_text = ", ".join(CODEBOOK)
        return (
            "\nNow answer questions about the ledger you already wrote.\n"
            + "\n".join(question_lines)
            + "\nI will provide each answer prefix one at a time.\n"
            + f"After each prefix, output exactly one word from this list and then stop the line: {codebook_text}\n"
            + "Do not write a sentence. Do not repeat the question. Do not add punctuation.\n"
            + "Correct example:\n"
            + "ANSWER 1: red\n"
            + "Incorrect examples:\n"
            + "ANSWER 1: The value in ENTRY 1 was red.\n"
            + "ANSWER 1: red.\n"
            + "ANSWER 1: ENTRY 1 was red\n"
        )
    answer_format = "\n".join(
        f"ANSWER {i + 1}: <value>"
        for i in range(len(query_positions))
    )
    return (
        "\nNow answer questions about the ledger you already wrote.\n"
        + "\n".join(question_lines)
        + "\nRespond with exactly these lines and nothing else:\n"
        + answer_format
        + "\n"
    )


def build_ledger_output_stub(num_entries: int):
    lines = [
        f"ENTRY {idx}: red"
        for idx in range(1, int(num_entries) + 1)
    ]
    lines.append("END LEDGER")
    return "\n".join(lines)


def build_answer_output_stub(num_queries: int):
    return "\n".join(
        f"ANSWER {idx}: red"
        for idx in range(1, int(num_queries) + 1)
    )


def choose_query_positions(num_entries: int, num_queries: int, rng):
    anchors = [4, int(num_entries) // 4, int(num_entries) // 2, max(1, int(num_entries) - 4)]
    while len(anchors) < int(num_queries):
        anchors.append(max(1, int(round((len(anchors) + 1) * int(num_entries) / float(int(num_queries) + 1)))))
    anchors = anchors[:int(num_queries)]
    positions = []
    for pos in anchors:
        jitter = rng.randint(-3, 3)
        positions.append(min(int(num_entries), max(1, int(pos) + jitter)))
    positions = sorted(set(positions))
    while len(positions) < int(num_queries):
        candidate = rng.randint(1, int(num_entries))
        if candidate not in positions:
            positions.append(candidate)
    return sorted(positions[:int(num_queries)])


def ensure_min_prompt_tokens(tokenizer, prompt: str, sample_idx: int, num_entries: int, filler_repeats: int, min_prompt_tokens: int):
    prompt_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    repeats = int(filler_repeats)
    while prompt_tokens < int(min_prompt_tokens):
        repeats += 8
        prompt = build_ledger_prompt(sample_idx, num_entries, repeats)
        prompt_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    return prompt, int(prompt_tokens), int(repeats)


def parse_entries(text: str):
    entries = {}
    for match in ENTRY_RE.finditer(text):
        idx = int(match.group(1))
        value = match.group(2).lower()
        if idx not in entries and value in CODEBOOK_SET:
            entries[idx] = value
    return entries


def parse_answers(text: str):
    answers = {}
    for match in ANSWER_RE.finditer(text):
        idx = int(match.group(1))
        value = match.group(2).lower()
        if idx not in answers and value in CODEBOOK_SET:
            answers[idx] = value
    return answers


def extract_generated_region(text: str):
    start = text.find("ENTRY 1:")
    if start < 0:
        return ""
    return text[start:]


def evaluate_output(text: str, query_positions, num_entries: int):
    region = extract_generated_region(text)
    entries = parse_entries(region)
    answers = parse_answers(region)
    hits = []
    for answer_idx, entry_idx in enumerate(query_positions, start=1):
        expected = entries.get(int(entry_idx))
        actual = answers.get(int(answer_idx))
        hits.append(bool(expected is not None and actual == expected))
    end_marker_present = "END LEDGER" in region
    format_ok = bool(len(entries) >= int(num_entries) and len(answers) >= len(query_positions) and end_marker_present)
    unique_entries = len(set(entries.values())) if entries else 0
    return {
        "entry_count": int(len(entries)),
        "answer_count": int(len(answers)),
        "query_hits": hits,
        "query_acc": float(np.mean(hits)) if hits else 0.0,
        "strict_acc": bool(len(hits) > 0 and all(hits)),
        "format_ok": format_ok,
        "end_marker_present": bool(end_marker_present),
        "unique_entries": int(unique_entries),
        "unique_entry_ratio": float(unique_entries / float(max(1, len(entries)))),
        "entries": entries,
        "answers": answers,
    }


def maybe_apply_chat_template(tokenizer, text: str, args):
    if not args.use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_text_config(config):
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def config_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def summarize_config(config):
    text_config = get_text_config(config)
    layer_types = config_get(text_config, "layer_types")
    layer_type_counts = None
    full_attention_layers = None
    linear_attention_layers = None
    if isinstance(layer_types, list):
        layer_type_counts = {
            str(kind): int(sum(1 for item in layer_types if str(item) == str(kind)))
            for kind in sorted({str(item) for item in layer_types})
        }
        full_attention_layers = [
            int(idx)
            for idx, kind in enumerate(layer_types)
            if str(kind) == "full_attention"
        ]
        linear_attention_layers = [
            int(idx)
            for idx, kind in enumerate(layer_types)
            if str(kind) == "linear_attention"
        ]
    fields = [
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "layer_types",
        "full_attention_interval",
        "vocab_size",
        "eos_token_id",
    ]
    return {
        "model_type": config_get(config, "model_type"),
        "architectures": config_get(config, "architectures"),
        "text_model_type": config_get(text_config, "model_type"),
        "text": {field: config_get(text_config, field) for field in fields if config_get(text_config, field) is not None},
        "attention_layer_plan_from_config": {
            "layer_type_counts": layer_type_counts,
            "full_attention_layers": full_attention_layers,
            "linear_attention_layers": linear_attention_layers,
            "ra_candidate_policy": "replace full_attention layers only; leave linear_attention/native non-softmax layers unchanged",
        },
        "top_level": {
            "image_token_id": config_get(config, "image_token_id"),
            "video_token_id": config_get(config, "video_token_id"),
            "vision_start_token_id": config_get(config, "vision_start_token_id"),
            "vision_end_token_id": config_get(config, "vision_end_token_id"),
        },
    }


def module_has_attrs(module, attrs):
    return all(hasattr(module, attr) for attr in attrs)


def classify_attention_module(name, module, layer_types):
    cls = type(module).__name__
    cls_lower = cls.lower()
    name_lower = name.lower()
    is_attention_module_name = (
        name_lower.endswith("self_attn")
        or name_lower.endswith("attention")
        or name_lower.endswith("linear_attention")
    )
    has_qkv = module_has_attrs(module, ("q_proj", "k_proj", "v_proj"))
    has_o = hasattr(module, "o_proj") or hasattr(module, "out_proj")
    has_linear_attention_markers = any(
        hasattr(module, attr)
        for attr in (
            "conv1d",
            "in_proj",
            "linear_q_proj",
            "linear_k_proj",
            "linear_v_proj",
            "dt_proj",
            "A_log",
        )
    )
    if "norm" in cls_lower or name_lower.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")):
        return None
    if not (
        has_qkv
        or has_linear_attention_markers
        or is_attention_module_name
        or "attention" in cls_lower
    ):
        return None

    layer_idx = None
    match = re.search(r"(?:layers|h|blocks)\.(\d+)", name)
    if match:
        layer_idx = int(match.group(1))
    config_layer_type = None
    if layer_idx is not None and isinstance(layer_types, list) and layer_idx < len(layer_types):
        config_layer_type = str(layer_types[layer_idx])

    if config_layer_type:
        kind = config_layer_type
    elif has_qkv and has_o:
        kind = "full_attention_candidate"
    elif "linear" in cls_lower or has_linear_attention_markers:
        kind = "linear_attention_candidate"
    else:
        kind = "attention_candidate"

    replaceable_full_attention = bool(
        has_qkv
        and has_o
        and (
            kind in {"full_attention", "full_attention_candidate"}
            or (config_layer_type is None and not has_linear_attention_markers)
        )
    )
    return {
        "name": name,
        "class": cls,
        "layer_idx": layer_idx,
        "kind": kind,
        "replaceable_full_attention": replaceable_full_attention,
        "has_q_proj": bool(hasattr(module, "q_proj")),
        "has_k_proj": bool(hasattr(module, "k_proj")),
        "has_v_proj": bool(hasattr(module, "v_proj")),
        "has_o_proj": bool(hasattr(module, "o_proj")),
        "num_heads": config_get(module, "num_heads", config_get(module, "num_attention_heads")),
        "num_key_value_heads": config_get(module, "num_key_value_heads"),
        "head_dim": config_get(module, "head_dim"),
    }


def inventory_model(model, config):
    text_config = get_text_config(config)
    layer_types = config_get(text_config, "layer_types", [])
    records = []
    for name, module in model.named_modules():
        row = classify_attention_module(name, module, layer_types)
        if row is not None:
            records.append(row)
    replaceable = [row for row in records if row.get("replaceable_full_attention")]
    skipped = [row for row in records if not row.get("replaceable_full_attention")]
    return {
        "config": summarize_config(config),
        "attention_modules": records,
        "counts_by_kind": {
            kind: sum(1 for row in records if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in records})
        },
        "replacement_plan": {
            "policy": "full_attention_modules_only",
            "replaceable_full_attention_count": int(len(replaceable)),
            "skipped_attention_like_count": int(len(skipped)),
            "target_module_names": [row["name"] for row in replaceable],
            "skipped_modules": [
                {
                    "name": row["name"],
                    "class": row["class"],
                    "kind": row["kind"],
                    "layer_idx": row["layer_idx"],
                }
                for row in skipped
            ],
        },
    }


def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    return modules.get(name)


def repeat_kv_for_attention(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if int(n_rep) == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        int(n_rep),
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * int(n_rep), seq_len, head_dim)


def module_num_attention_heads(module):
    value = config_get(module, "num_attention_heads")
    if value is None:
        value = config_get(module, "num_heads")
    if value is None:
        value = config_get(getattr(module, "config", None), "num_attention_heads")
    if value is None:
        value = config_get(getattr(module, "config", None), "num_heads")
    if value is None:
        value = int(module.num_key_value_groups) * int(module_num_key_value_heads(module))
    return int(value)


def module_num_key_value_heads(module):
    value = config_get(module, "num_key_value_heads")
    if value is None:
        value = config_get(getattr(module, "config", None), "num_key_value_heads")
    if value is None:
        value = max(1, module_num_attention_heads(module) // int(module.num_key_value_groups))
    return int(value)


def project_qkv_states(module, hidden_states):
    input_shape = hidden_states.shape[:-1]
    head_dim = int(module.head_dim)
    hidden_shape = (*input_shape, -1, head_dim)

    q_projected = module.q_proj(hidden_states)
    gate = None
    # Qwen3.5 full-attention uses a fused q/gate projection:
    # [batch, seq, heads, 2 * head_dim] -> query + output gate.
    if (
        hasattr(module, "q_norm")
        and hasattr(module, "k_norm")
        and int(q_projected.shape[-1]) % int(2 * head_dim) == 0
    ):
        query_states, gate = torch.chunk(
            q_projected.view(*input_shape, -1, int(2 * head_dim)),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
    else:
        query_states = q_projected.view(hidden_shape)

    key_states = module.k_proj(hidden_states).view(hidden_shape)
    value_states = module.v_proj(hidden_states).view(hidden_shape)

    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    if hasattr(module, "k_norm"):
        key_states = module.k_norm(key_states)

    return (
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        gate,
    )


def apply_attention_output_projection(module, attn_output, input_shape, gate=None):
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    return module.o_proj(attn_output)


def original_forward_accepts(original_forward, parameter_name: str):
    try:
        return parameter_name in inspect.signature(original_forward).parameters
    except (TypeError, ValueError):
        return False


def call_original_attention_forward(
    original_forward,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values,
    cache_position,
    kwargs,
):
    call_kwargs = dict(kwargs)
    if original_forward_accepts(original_forward, "past_key_values"):
        call_kwargs["past_key_values"] = past_key_values
    else:
        call_kwargs["past_key_value"] = past_key_values
    if original_forward_accepts(original_forward, "cache_position"):
        call_kwargs["cache_position"] = cache_position
    return original_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        **call_kwargs,
    )


def update_attention_cache(past_key_values, key_states, value_states, module, cos, sin, cache_position):
    if past_key_values is None:
        return key_states, value_states
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    try:
        return past_key_values.update(
            key_states,
            value_states,
            module.layer_idx,
            cache_kwargs,
        )
    except TypeError:
        return past_key_values.update(
            key_states,
            value_states,
            module.layer_idx,
        )


def dense_scores_and_repeated_values(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling: float,
):
    key_states = repeat_kv_for_attention(key_states, int(module.num_key_value_groups))
    value_states = repeat_kv_for_attention(value_states, int(module.num_key_value_groups))
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    return scores, value_states


def sparse_attention_from_keep(scores, value_states, keep, query_dtype):
    masked_scores = scores.masked_fill(~keep[:, :, None, :], torch.finfo(scores.dtype).min)
    attn_weights = torch.nn.functional.softmax(masked_scores, dim=-1, dtype=torch.float32).to(query_dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    kept = int(keep.sum().item())
    total = int(keep.numel())
    return attn_output, attn_weights, kept, total


def oracle_topk_keep_mask(scores, topk: int, static_prefix: int, static_suffix: int):
    batch, heads, q_len, kv_len = scores.shape
    if q_len != 1:
        raise RuntimeError("oracle_topk attention is intended for decode q_len=1 only.")

    keep = torch.zeros((batch, heads, kv_len), dtype=torch.bool, device=scores.device)
    prefix_end = min(max(0, int(static_prefix)), int(kv_len))
    suffix_start = max(prefix_end, int(kv_len) - max(0, int(static_suffix)))
    if prefix_end > 0:
        keep[:, :, :prefix_end] = True
    if suffix_start < kv_len:
        keep[:, :, suffix_start:] = True

    dyn_start = prefix_end
    dyn_end = suffix_start
    dyn_len = max(0, int(dyn_end) - int(dyn_start))
    k = min(max(0, int(topk)), dyn_len)
    if k > 0:
        dyn_scores = scores[:, :, 0, dyn_start:dyn_end]
        top_idx = torch.topk(dyn_scores, k=k, dim=-1).indices + int(dyn_start)
        keep.scatter_(dim=-1, index=top_idx, value=True)

    if not bool(keep.any()):
        keep[:, :, -1:] = True
    return keep


def oracle_topk_attention_forward(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling: float,
    topk: int,
    static_prefix: int,
    static_suffix: int,
):
    scores, repeated_values = dense_scores_and_repeated_values(
        module=module,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        scaling=scaling,
    )
    keep = oracle_topk_keep_mask(
        scores=scores,
        topk=topk,
        static_prefix=static_prefix,
        static_suffix=static_suffix,
    )
    return sparse_attention_from_keep(
        scores=scores,
        value_states=repeated_values,
        keep=keep,
        query_dtype=query_states.dtype,
    )


class HFOracleTopKPatcher:
    def __init__(self, model, inventory, topk: int, static_prefix: int, static_suffix: int):
        self.model = model
        self.inventory = inventory
        self.topk = int(topk)
        self.static_prefix = int(static_prefix)
        self.static_suffix = int(static_suffix)
        self.records = []
        self.original_forwards = {}
        self.decode_calls = 0
        self.prefill_passthrough_calls = 0
        self.kept_total = 0
        self.keep_space_total = 0

    def _wrap_module(self, name, module):
        original_forward = module.forward
        forward_globals = getattr(original_forward, "__globals__", None)
        if forward_globals is None and hasattr(original_forward, "__func__"):
            forward_globals = getattr(original_forward.__func__, "__globals__", None)
        apply_rotary_pos_emb = (forward_globals or {}).get("apply_rotary_pos_emb")
        if apply_rotary_pos_emb is None:
            self.records.append(
                {
                    "name": name,
                    "class": type(module).__name__,
                    "installed": False,
                    "reason": "apply_rotary_pos_emb_not_found",
                }
            )
            return

        patcher = self

        def wrapped_forward(
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            past_key_values = kwargs.pop("past_key_values", past_key_value)
            if hidden_states.shape[1] != 1:
                patcher.prefill_passthrough_calls += 1
                return call_original_attention_forward(
                    original_forward=original_forward,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    kwargs=kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            query_states, key_states, value_states, gate = project_qkv_states(module, hidden_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states, value_states = update_attention_cache(
                past_key_values=past_key_values,
                key_states=key_states,
                value_states=value_states,
                module=module,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
            )

            attn_output, attn_weights, kept, total = oracle_topk_attention_forward(
                module=module,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                scaling=float(module.scaling),
                topk=patcher.topk,
                static_prefix=patcher.static_prefix,
                static_suffix=patcher.static_suffix,
            )
            patcher.decode_calls += 1
            patcher.kept_total += int(kept)
            patcher.keep_space_total += int(total)
            attn_output = apply_attention_output_projection(module, attn_output, input_shape, gate=gate)
            return attn_output, attn_weights if kwargs.get("output_attentions", False) else None

        module.forward = wrapped_forward
        self.original_forwards[name] = original_forward
        self.records.append(
            {
                "name": name,
                "class": type(module).__name__,
                "installed": True,
                "layer_idx": getattr(module, "layer_idx", None),
            }
        )

    def install(self):
        target_names = self.inventory.get("replacement_plan", {}).get("target_module_names", [])
        for name in target_names:
            module = get_module_by_name(self.model, name)
            if module is None:
                self.records.append(
                    {
                        "name": name,
                        "installed": False,
                        "reason": "module_not_found",
                    }
                )
                continue
            self._wrap_module(name, module)
        return self

    def summary(self):
        installed = [row for row in self.records if row.get("installed")]
        keep_ratio = (
            float(self.kept_total) / float(self.keep_space_total)
            if self.keep_space_total > 0
            else None
        )
        return {
            "mode": "oracle_topk",
            "topk": int(self.topk),
            "static_prefix": int(self.static_prefix),
            "static_suffix": int(self.static_suffix),
            "installed_count": int(len(installed)),
            "records": self.records,
            "decode_calls": int(self.decode_calls),
            "prefill_passthrough_calls": int(self.prefill_passthrough_calls),
            "kept_total": int(self.kept_total),
            "keep_space_total": int(self.keep_space_total),
            "keep_ratio": keep_ratio,
        }


class HFGraphTopKModuleState:
    def __init__(self):
        self.invalid = True
        self.base_len = 0
        self.last_attached_len = 0
        self.graph = []
        self.hubs = []
        self.extra_edges = []
        self.pending_edges = []
        self.prev_seeds = []


class HFGraphTopKPatcher:
    def __init__(
        self,
        model,
        inventory,
        topk: int,
        static_prefix: int,
        static_suffix: int,
        graph_degree: int,
        visit_budget: int,
        seed_count: int,
        online_edges: int,
    ):
        self.model = model
        self.inventory = inventory
        self.topk = max(1, int(topk))
        self.static_prefix = max(0, int(static_prefix))
        self.static_suffix = max(0, int(static_suffix))
        self.graph_degree = max(1, int(graph_degree))
        self.visit_budget = max(1, int(visit_budget))
        self.seed_count = max(1, int(seed_count))
        self.online_edges = max(0, int(online_edges))
        self.records = []
        self.original_forwards = {}
        self.module_states = {}
        self.decode_calls = 0
        self.prefill_passthrough_calls = 0
        self.graph_builds = 0
        self.visited_total = 0
        self.candidates_total = 0
        self.online_edges_total = 0
        self.kept_total = 0
        self.keep_space_total = 0

    def _dynamic_bounds(self, kv_len: int):
        prefix_end = min(self.static_prefix, int(kv_len))
        suffix_start = max(prefix_end, int(kv_len) - self.static_suffix)
        return int(prefix_end), int(suffix_start)

    def _get_state(self, name, module):
        state = self.module_states.get(name)
        if state is None:
            state = HFGraphTopKModuleState()
            state.pending_edges = [[] for _ in range(module_num_key_value_heads(module))]
            state.prev_seeds = [[] for _ in range(module_num_attention_heads(module))]
            self.module_states[name] = state
        return state

    def _build_graph(self, state, key_states, module):
        # key_states is [bs, kv_heads, kv_len, head_dim] after HF cache update.
        kv_heads = int(key_states.shape[1])
        kv_len = int(key_states.shape[2])
        graph_start = min(self.static_prefix, kv_len)
        state.graph = []
        state.hubs = []
        state.extra_edges = [dict() for _ in range(kv_heads)]
        for kv_hdx in range(kv_heads):
            adjacency = [[] for _ in range(kv_len)]
            if graph_start < kv_len:
                rows = key_states[0, kv_hdx, graph_start:kv_len, :].detach().float()
                node_count = int(rows.shape[0])
                if node_count > 1:
                    scores = torch.matmul(rows, rows.transpose(0, 1))
                    scores.fill_diagonal_(torch.finfo(scores.dtype).min)
                    degree = min(self.graph_degree, node_count - 1)
                    neighbors = torch.topk(scores, k=degree, dim=-1).indices.detach().cpu().tolist()
                    for local_idx, row in enumerate(neighbors):
                        adjacency[graph_start + local_idx] = [graph_start + int(col) for col in row]
                norms = torch.linalg.vector_norm(rows, dim=-1)
                hub_k = min(self.seed_count, int(norms.numel()))
                if hub_k > 0:
                    hubs = (torch.topk(norms, k=hub_k).indices + graph_start).detach().cpu().tolist()
                else:
                    hubs = []
            else:
                hubs = []
            state.graph.append(adjacency)
            state.hubs.append([int(x) for x in hubs])
        state.base_len = kv_len
        state.last_attached_len = kv_len
        state.pending_edges = [[] for _ in range(kv_heads)]
        state.prev_seeds = [[] for _ in range(module_num_attention_heads(module))]
        state.invalid = False
        self.graph_builds += 1

    def _ensure_graph(self, state, key_states, module):
        if state.invalid or not state.graph:
            self._build_graph(state, key_states, module)

    def _neighbors(self, state, kv_hdx: int, node: int):
        out = []
        if 0 <= int(kv_hdx) < len(state.graph):
            graph = state.graph[int(kv_hdx)]
            if 0 <= int(node) < len(graph):
                out.extend(graph[int(node)])
        if 0 <= int(kv_hdx) < len(state.extra_edges):
            out.extend(state.extra_edges[int(kv_hdx)].get(int(node), ()))
        return out

    def _attach_new_nodes(self, state, kv_len: int):
        if self.online_edges <= 0:
            state.last_attached_len = int(kv_len)
            return
        for pos in range(int(state.last_attached_len), int(kv_len)):
            for kv_hdx in range(len(state.extra_edges)):
                pending = list(state.pending_edges[kv_hdx])[: self.online_edges]
                if not pending:
                    pending = list(state.hubs[kv_hdx])[: self.online_edges]
                if not pending:
                    continue
                edge_map = state.extra_edges[kv_hdx]
                pos_edges = edge_map.setdefault(int(pos), set())
                for neighbor in pending:
                    neighbor = int(neighbor)
                    if neighbor == int(pos) or neighbor < 0 or neighbor >= int(kv_len):
                        continue
                    pos_edges.add(neighbor)
                    edge_map.setdefault(neighbor, set()).add(int(pos))
                    self.online_edges_total += 2
        state.pending_edges = [[] for _ in range(len(state.extra_edges))]
        state.last_attached_len = int(kv_len)

    def _before_graph_keep(self, state, query_states, key_states, module):
        del state, query_states, key_states, module

    def _seed_nodes(self, state, hdx: int, kv_hdx: int, kv_len: int, dyn_start: int, dyn_end: int):
        seeds = []
        seeds.extend(state.prev_seeds[hdx][: self.seed_count])
        seeds.extend(state.hubs[kv_hdx][: self.seed_count])
        if dyn_end > dyn_start:
            tail_start = max(dyn_start, dyn_end - self.seed_count)
            seeds.extend(range(tail_start, dyn_end))
            seeds.append(dyn_start)
        filtered = []
        seen = set()
        for seed in seeds:
            seed = int(seed)
            if seed < dyn_start or seed >= kv_len or seed in seen:
                continue
            filtered.append(seed)
            seen.add(seed)
            if len(filtered) >= self.seed_count * 3:
                break
        return filtered

    def _graph_keep_mask(self, state, scores, module):
        batch, heads, q_len, kv_len = scores.shape
        if batch != 1 or q_len != 1:
            raise RuntimeError("graph_topk attention currently supports batch=1 and decode q_len=1 only.")
        dyn_start, dyn_end = self._dynamic_bounds(kv_len)
        keep = torch.zeros((batch, heads, kv_len), dtype=torch.bool, device=scores.device)
        if dyn_start > 0:
            keep[:, :, :dyn_start] = True
        if dyn_end < kv_len:
            keep[:, :, dyn_end:] = True

        selected_by_kv = [[] for _ in range(module_num_key_value_heads(module))]
        for hdx in range(int(heads)):
            kv_hdx = int(hdx) // int(module.num_key_value_groups)
            seeds = self._seed_nodes(state, hdx, kv_hdx, kv_len, dyn_start, dyn_end)
            if not seeds:
                continue
            # Avoid per-candidate GPU synchronizations from Tensor.item() inside
            # Python graph traversal. The graph prototype is CPU/Python anyway,
            # so copy this head's score row once and use CPU scalars below.
            head_scores = scores[0, hdx, 0, :].detach().float().cpu()

            frontier = []
            visited = set()
            candidates = []
            for seed in seeds:
                score = float(head_scores[int(seed)])
                frontier.append((-score, int(seed)))
            import heapq

            heapq.heapify(frontier)
            while frontier and len(visited) < self.visit_budget:
                neg_score, node = heapq.heappop(frontier)
                if node in visited:
                    continue
                visited.add(node)
                score = -float(neg_score)
                candidates.append((score, int(node)))
                for neighbor in self._neighbors(state, kv_hdx, node):
                    neighbor = int(neighbor)
                    if neighbor in visited or neighbor < dyn_start or neighbor >= kv_len:
                        continue
                    next_score = float(head_scores[neighbor])
                    heapq.heappush(frontier, (-next_score, neighbor))

            self.visited_total += int(len(visited))
            self.candidates_total += int(len(candidates))
            dynamic_candidates = [
                (score, node)
                for score, node in candidates
                if dyn_start <= int(node) < dyn_end
            ]
            dynamic_candidates.sort(reverse=True, key=lambda item: item[0])
            selected = [int(node) for _score, node in dynamic_candidates[: self.topk]]
            if selected:
                idx = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
                keep[0, hdx].scatter_(dim=-1, index=idx, value=True)
                state.prev_seeds[hdx] = selected[: self.seed_count]
                selected_by_kv[kv_hdx].extend(selected[: self.online_edges])

        for kv_hdx, selected in enumerate(selected_by_kv):
            if not selected:
                continue
            deduped = []
            seen = set()
            for token_idx in selected:
                token_idx = int(token_idx)
                if token_idx in seen:
                    continue
                deduped.append(token_idx)
                seen.add(token_idx)
                if len(deduped) >= self.online_edges:
                    break
            state.pending_edges[kv_hdx] = deduped

        if not bool(keep.any()):
            keep[:, :, -1:] = True
        return keep

    def _wrap_module(self, name, module):
        original_forward = module.forward
        forward_globals = getattr(original_forward, "__globals__", None)
        if forward_globals is None and hasattr(original_forward, "__func__"):
            forward_globals = getattr(original_forward.__func__, "__globals__", None)
        apply_rotary_pos_emb = (forward_globals or {}).get("apply_rotary_pos_emb")
        if apply_rotary_pos_emb is None:
            self.records.append(
                {
                    "name": name,
                    "class": type(module).__name__,
                    "installed": False,
                    "reason": "apply_rotary_pos_emb_not_found",
                }
            )
            return

        patcher = self
        state = self._get_state(name, module)

        def wrapped_forward(
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            past_key_values = kwargs.pop("past_key_values", past_key_value)
            if hidden_states.shape[1] != 1:
                patcher.prefill_passthrough_calls += 1
                state.invalid = True
                return call_original_attention_forward(
                    original_forward=original_forward,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    kwargs=kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            query_states, key_states, value_states, gate = project_qkv_states(module, hidden_states)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states, value_states = update_attention_cache(
                past_key_values=past_key_values,
                key_states=key_states,
                value_states=value_states,
                module=module,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
            )

            patcher._ensure_graph(state, key_states, module)
            patcher._attach_new_nodes(state, int(key_states.shape[2]))
            scores, repeated_values = dense_scores_and_repeated_values(
                module=module,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                scaling=float(module.scaling),
            )
            patcher._before_graph_keep(state, query_states, key_states, module)
            keep = patcher._graph_keep_mask(state, scores, module)
            attn_output, attn_weights, kept, total = sparse_attention_from_keep(
                scores=scores,
                value_states=repeated_values,
                keep=keep,
                query_dtype=query_states.dtype,
            )
            patcher.decode_calls += 1
            patcher.kept_total += int(kept)
            patcher.keep_space_total += int(total)
            attn_output = apply_attention_output_projection(module, attn_output, input_shape, gate=gate)
            return attn_output, attn_weights if kwargs.get("output_attentions", False) else None

        module.forward = wrapped_forward
        self.original_forwards[name] = original_forward
        self.records.append(
            {
                "name": name,
                "class": type(module).__name__,
                "installed": True,
                "layer_idx": getattr(module, "layer_idx", None),
            }
        )

    def install(self):
        target_names = self.inventory.get("replacement_plan", {}).get("target_module_names", [])
        for name in target_names:
            module = get_module_by_name(self.model, name)
            if module is None:
                self.records.append(
                    {
                        "name": name,
                        "installed": False,
                        "reason": "module_not_found",
                    }
                )
                continue
            self._wrap_module(name, module)
        return self

    def summary(self):
        installed = [row for row in self.records if row.get("installed")]
        keep_ratio = (
            float(self.kept_total) / float(self.keep_space_total)
            if self.keep_space_total > 0
            else None
        )
        return {
            "mode": "graph_topk",
            "topk": int(self.topk),
            "static_prefix": int(self.static_prefix),
            "static_suffix": int(self.static_suffix),
            "graph_degree": int(self.graph_degree),
            "visit_budget": int(self.visit_budget),
            "seed_count": int(self.seed_count),
            "online_edges": int(self.online_edges),
            "installed_count": int(len(installed)),
            "records": self.records,
            "decode_calls": int(self.decode_calls),
            "prefill_passthrough_calls": int(self.prefill_passthrough_calls),
            "graph_builds": int(self.graph_builds),
            "visited_total": int(self.visited_total),
            "candidates_total": int(self.candidates_total),
            "online_edges_total": int(self.online_edges_total),
            "kept_total": int(self.kept_total),
            "keep_space_total": int(self.keep_space_total),
            "keep_ratio": keep_ratio,
        }


class HFGraphTopKRoarModuleState(HFGraphTopKModuleState):
    def __init__(self):
        super().__init__()
        self.base_offsets = []
        self.base_neighbors = []
        self.csr_offsets = []
        self.csr_neighbors = []
        self.csr_len = []
        self.csr_dirty = []
        self.csr_offsets_cuda = []
        self.csr_neighbors_cuda = []
        self.csr_cuda_len = []
        self.csr_cuda_device = []
        self.csr_cuda_dirty = []
        self.base_offsets_cuda = []
        self.base_neighbors_cuda = []
        self.base_csr_cuda_capacity = []
        self.base_csr_cuda_len = []
        self.base_csr_cuda_device = []
        self.overlay_counts_cuda = []
        self.overlay_neighbors_cuda = []
        self.overlay_cuda_capacity = []
        self.overlay_cuda_width = []
        self.overlay_cuda_device = []
        self.overlay_dirty = []
        self.keys_cpu = []
        self.keys_cuda = []
        self.keys_cuda_storage = []
        self.keys_cuda_capacity = []
        self.keys_cuda_len = []
        self.keys_cuda_device = []
        self.hub_seed_ids_cuda = []
        self.hub_seed_ids_cuda_device = ""
        self.prev_seed_ids_cuda = None
        self.prev_seed_counts_cuda = None
        self.prev_seed_width = 0
        self.prev_seed_device = ""
        self.current_query_states = None
        self.current_key_states = None


class HFGraphTopKRoarPatcher(HFGraphTopKPatcher):
    def __init__(
        self,
        model,
        inventory,
        topk: int,
        static_prefix: int,
        static_suffix: int,
        graph_degree: int,
        visit_budget: int,
        seed_count: int,
        online_edges: int,
        search_backend: str,
        candidate_target: int,
        expand_width: int,
        min_visits: int,
        frontier_topn: int,
        stop_patience: int,
        stop_margin: float,
        roar_cand_limit: int,
        roar_enhance_limit: int,
        roar_entry: str,
        roar_threads: int,
    ):
        super().__init__(
            model=model,
            inventory=inventory,
            topk=topk,
            static_prefix=static_prefix,
            static_suffix=static_suffix,
            graph_degree=graph_degree,
            visit_budget=visit_budget,
            seed_count=seed_count,
            online_edges=online_edges,
        )
        self.search_backend = str(search_backend)
        self.candidate_target = max(int(self.topk), int(candidate_target) if int(candidate_target) > 0 else int(self.topk))
        self.expand_width = max(1, int(expand_width))
        self.min_visits = max(1, int(min_visits))
        self.frontier_topn = max(0, int(frontier_topn))
        self.stop_patience = max(0, int(stop_patience))
        self.stop_margin = float(stop_margin)
        self.roar_cand_limit = max(1, int(roar_cand_limit))
        self.roar_enhance_limit = max(1, int(roar_enhance_limit))
        self.roar_entry = str(roar_entry)
        self.roar_threads = max(0, int(roar_threads))
        self.roar = load_roar_backend_module()
        if not self.roar.roargraph_cpp_available():
            raise RuntimeError(
                "graph_topk_roar requires the RoarGraph C++ extension. "
                f"Import error: {self.roar.roargraph_cpp_import_error()}"
            )
        self.cuda_fallbacks = 0
        self.fullgpu_fallback_reasons = collections.Counter()
        self.csr_cuda_uploads = 0
        self.base_csr_cuda_uploads = 0
        self.overlay_cuda_uploads = 0
        self.key_cuda_uploads = 0
        self.key_cuda_append_updates = 0
        self.prev_seed_cuda_uploads = 0
        self.stop_reasons = collections.Counter()

    def _get_state(self, name, module):
        state = self.module_states.get(name)
        if state is None:
            state = HFGraphTopKRoarModuleState()
            state.pending_edges = [[] for _ in range(module_num_key_value_heads(module))]
            state.prev_seeds = [[] for _ in range(module_num_attention_heads(module))]
            self.module_states[name] = state
        return state

    def _ensure_fullgpu_seed_tensors(self, state, module, device, *, kv_heads=None, heads=None):
        kv_heads = module_num_key_value_heads(module) if kv_heads is None else int(kv_heads)
        heads = module_num_attention_heads(module) if heads is None else int(heads)
        hub_cap = max(1, min(128, int(self.seed_count)))
        prev_width = max(1, min(512, int(self.seed_count)))
        device_key = str(device)

        hub_ok = (
            state.hub_seed_ids_cuda
            and len(state.hub_seed_ids_cuda) == kv_heads
            and state.hub_seed_ids_cuda_device == device_key
        )
        if not hub_ok:
            state.hub_seed_ids_cuda = []
            for kv_hdx in range(kv_heads):
                seeds = state.hubs[kv_hdx][:hub_cap] if kv_hdx < len(state.hubs) else []
                if seeds:
                    state.hub_seed_ids_cuda.append(torch.as_tensor(seeds, dtype=torch.int32, device=device))
                else:
                    state.hub_seed_ids_cuda.append(torch.empty((0,), dtype=torch.int32, device=device))
            state.hub_seed_ids_cuda_device = device_key

        prev_ok = (
            state.prev_seed_ids_cuda is not None
            and state.prev_seed_counts_cuda is not None
            and int(state.prev_seed_width) == prev_width
            and state.prev_seed_device == device_key
            and int(state.prev_seed_ids_cuda.shape[0]) == heads
        )
        if not prev_ok:
            state.prev_seed_ids_cuda = torch.full(
                (heads, prev_width),
                -1,
                dtype=torch.int32,
                device=device,
            )
            state.prev_seed_counts_cuda = torch.zeros((heads,), dtype=torch.int32, device=device)
            state.prev_seed_width = int(prev_width)
            state.prev_seed_device = device_key
            self.prev_seed_cuda_uploads += 1
        return prev_width

    def _update_key_cuda_cache(self, state, kv_hdx: int, keys_t: torch.Tensor, kv_len: int):
        kv_hdx = int(kv_hdx)
        kv_len = int(kv_len)
        device = keys_t.device
        device_key = str(device)
        head_dim = int(keys_t.shape[-1])
        while kv_hdx >= len(state.keys_cuda_storage):
            state.keys_cuda_storage.append(None)
            state.keys_cuda_capacity.append(0)
            state.keys_cuda_len.append(0)
            state.keys_cuda_device.append("")
        while kv_hdx >= len(state.keys_cuda):
            state.keys_cuda.append(None)

        storage = state.keys_cuda_storage[kv_hdx]
        capacity = int(state.keys_cuda_capacity[kv_hdx])
        cached_len = int(state.keys_cuda_len[kv_hdx])
        cache_ok = (
            storage is not None
            and int(storage.shape[-1]) == head_dim
            and capacity >= kv_len
            and state.keys_cuda_device[kv_hdx] == device_key
        )
        if not cache_ok or cached_len > kv_len:
            capacity = self._next_capacity(kv_len)
            storage = torch.empty((capacity, head_dim), dtype=torch.float32, device=device)
            if kv_len > 0:
                storage[:kv_len].copy_(keys_t[:kv_len].float(), non_blocking=True)
            state.keys_cuda_storage[kv_hdx] = storage
            state.keys_cuda_capacity[kv_hdx] = int(capacity)
            state.keys_cuda_device[kv_hdx] = device_key
            state.keys_cuda_len[kv_hdx] = int(kv_len)
            state.keys_cuda[kv_hdx] = storage[:kv_len]
            self.key_cuda_uploads += 1
            return

        if cached_len < kv_len:
            storage[cached_len:kv_len].copy_(keys_t[cached_len:kv_len].float(), non_blocking=True)
            state.keys_cuda_len[kv_hdx] = int(kv_len)
            self.key_cuda_append_updates += 1
        state.keys_cuda[kv_hdx] = storage[:kv_len]

    def _empty_csr(self, kv_len: int):
        return (
            np.zeros((int(kv_len) + 1,), dtype=np.uint32),
            np.empty((0,), dtype=np.int32),
        )

    def _exact_dynamic_knn(self, keys_2d: torch.Tensor, dyn_start: int, dyn_end: int):
        node_count = max(0, int(dyn_end) - int(dyn_start))
        if node_count <= 1:
            return np.empty((0, 0), dtype=np.int32)
        k = min(max(1, int(self.graph_degree)), node_count - 1)
        base = keys_2d[int(dyn_start):int(dyn_end)].detach().float().contiguous()
        out = torch.empty((node_count, k), dtype=torch.int32, device="cpu")
        chunk = 128
        for q_start in range(0, node_count, chunk):
            q_end = min(node_count, q_start + chunk)
            q = base[q_start:q_end]
            scores = torch.matmul(q, base.transpose(0, 1))
            local = torch.arange(q_start, q_end, device=scores.device)
            scores[torch.arange(q_end - q_start, device=scores.device), local] = torch.finfo(scores.dtype).min
            idx = torch.topk(scores, k=k, dim=-1, sorted=False).indices.to(torch.int32)
            out[q_start:q_end].copy_((idx + int(dyn_start)).cpu())
        return np.ascontiguousarray(out.numpy(), dtype=np.int32)

    def _build_graph(self, state, key_states, module):
        kv_heads = int(key_states.shape[1])
        kv_len = int(key_states.shape[2])
        dyn_start, dyn_end = self._dynamic_bounds(kv_len)
        state.base_offsets = []
        state.base_neighbors = []
        state.csr_offsets = []
        state.csr_neighbors = []
        state.csr_len = []
        state.csr_dirty = []
        state.csr_offsets_cuda = []
        state.csr_neighbors_cuda = []
        state.csr_cuda_len = []
        state.csr_cuda_device = []
        state.csr_cuda_dirty = []
        state.base_offsets_cuda = []
        state.base_neighbors_cuda = []
        state.base_csr_cuda_capacity = []
        state.base_csr_cuda_len = []
        state.base_csr_cuda_device = []
        state.overlay_counts_cuda = []
        state.overlay_neighbors_cuda = []
        state.overlay_cuda_capacity = []
        state.overlay_cuda_width = []
        state.overlay_cuda_device = []
        state.overlay_dirty = []
        state.keys_cpu = []
        state.keys_cuda = []
        state.keys_cuda_storage = []
        state.keys_cuda_capacity = []
        state.keys_cuda_len = []
        state.keys_cuda_device = []
        state.hub_seed_ids_cuda = []
        state.hub_seed_ids_cuda_device = ""
        state.prev_seed_ids_cuda = None
        state.prev_seed_counts_cuda = None
        state.prev_seed_width = 0
        state.prev_seed_device = ""
        state.graph = [[] for _ in range(kv_heads)]
        state.hubs = []
        state.extra_edges = [dict() for _ in range(kv_heads)]

        for kv_hdx in range(kv_heads):
            keys_t = key_states[0, kv_hdx, :kv_len, :].detach().contiguous()
            keys_cpu = np.ascontiguousarray(keys_t.float().cpu().numpy(), dtype=np.float32)
            state.keys_cpu.append(keys_cpu)
            state.keys_cuda.append(None)
            state.keys_cuda_storage.append(None)
            state.keys_cuda_capacity.append(0)
            state.keys_cuda_len.append(0)
            state.keys_cuda_device.append("")
            if keys_t.device.type == "cuda" and self.search_backend in {"cuda_group", "cuda_fullgpu"}:
                self._update_key_cuda_cache(state, kv_hdx, keys_t, kv_len)

            if dyn_end > dyn_start:
                dyn_rows = keys_t[int(dyn_start):int(dyn_end)].float()
                norms = torch.linalg.vector_norm(dyn_rows, dim=-1)
                hub_k = min(self.seed_count, int(norms.numel()))
                hubs = (torch.topk(norms, k=hub_k).indices + int(dyn_start)).detach().cpu().tolist() if hub_k > 0 else []
            else:
                hubs = []
            state.hubs.append([int(x) for x in hubs])

            if dyn_end - dyn_start > 1:
                knn = self._exact_dynamic_knn(keys_t, dyn_start, dyn_end)
                offsets, neighbors, _meta = self.roar.build_roar_graph_csr_cpp(
                    knn,
                    keys_cpu,
                    dynamic_start=int(dyn_start),
                    dynamic_end=int(dyn_end),
                    nq=min(max(1, int(self.graph_degree)), int(knn.shape[1]) if knn.ndim == 2 and knn.shape[1] > 0 else 1),
                    degree_cap=int(self.graph_degree),
                    cand_limit=int(self.roar_cand_limit),
                    enable_enhance=True,
                    enhance_limit=int(self.roar_enhance_limit),
                    entry_mode=str(self.roar_entry),
                    max_query_per_pivot=0,
                    num_threads=int(self.roar_threads),
                )
                offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
                neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
            else:
                offsets, neighbors = self._empty_csr(kv_len)

            state.base_offsets.append(offsets)
            state.base_neighbors.append(neighbors)
            state.csr_offsets.append(offsets)
            state.csr_neighbors.append(neighbors)
            state.csr_len.append(kv_len)
            state.csr_dirty.append(False)
            state.csr_offsets_cuda.append(None)
            state.csr_neighbors_cuda.append(None)
            state.csr_cuda_len.append(-1)
            state.csr_cuda_device.append("")
            state.csr_cuda_dirty.append(True)
            state.base_offsets_cuda.append(None)
            state.base_neighbors_cuda.append(None)
            state.base_csr_cuda_capacity.append(0)
            state.base_csr_cuda_len.append(0)
            state.base_csr_cuda_device.append("")
            state.overlay_counts_cuda.append(None)
            state.overlay_neighbors_cuda.append(None)
            state.overlay_cuda_capacity.append(0)
            state.overlay_cuda_width.append(0)
            state.overlay_cuda_device.append("")
            state.overlay_dirty.append(True)

        state.base_len = kv_len
        state.last_attached_len = kv_len
        state.pending_edges = [[] for _ in range(kv_heads)]
        state.prev_seeds = [[] for _ in range(module_num_attention_heads(module))]
        if self.search_backend == "cuda_fullgpu" and key_states.device.type == "cuda":
            self._ensure_fullgpu_seed_tensors(state, module, key_states.device)
        state.invalid = False
        self.graph_builds += 1

    def _before_graph_keep(self, state, query_states, key_states, module):
        del module
        state.current_query_states = query_states.detach()
        state.current_key_states = key_states.detach()
        kv_len = int(key_states.shape[2])
        for kv_hdx in range(int(key_states.shape[1])):
            keys_t = key_states[0, kv_hdx, :kv_len, :].detach()
            if keys_t.device.type == "cuda" and self.search_backend in {"cuda_group", "cuda_fullgpu"}:
                self._update_key_cuda_cache(state, kv_hdx, keys_t, kv_len)
            if kv_hdx >= len(state.keys_cpu) or state.keys_cpu[kv_hdx].shape[0] != kv_len:
                while kv_hdx >= len(state.keys_cpu):
                    state.keys_cpu.append(np.empty((0, int(keys_t.shape[-1])), dtype=np.float32))
                    state.keys_cuda.append(None)
                    state.keys_cuda_storage.append(None)
                    state.keys_cuda_capacity.append(0)
                    state.keys_cuda_len.append(0)
                    state.keys_cuda_device.append("")
                if self.search_backend == "cpp":
                    state.keys_cpu[kv_hdx] = np.ascontiguousarray(keys_t.float().cpu().numpy(), dtype=np.float32)
                    if keys_t.device.type == "cuda":
                        state.keys_cuda[kv_hdx] = keys_t.float().contiguous()
                if self.search_backend != "cuda_fullgpu" and kv_hdx < len(state.csr_dirty):
                    state.csr_dirty[kv_hdx] = True
                if self.search_backend != "cuda_fullgpu" and kv_hdx < len(state.csr_cuda_dirty):
                    state.csr_cuda_dirty[kv_hdx] = True
                if self.search_backend != "cuda_fullgpu" and kv_hdx < len(state.overlay_dirty):
                    state.overlay_dirty[kv_hdx] = True

    def _attach_new_nodes(self, state, kv_len: int):
        before = int(self.online_edges_total)
        prev_len = int(state.last_attached_len)
        super()._attach_new_nodes(state, kv_len)
        if int(kv_len) != prev_len or int(self.online_edges_total) != before:
            if self.search_backend != "cuda_fullgpu":
                for i in range(len(state.csr_dirty)):
                    state.csr_dirty[i] = True
                for i in range(len(state.csr_cuda_dirty)):
                    state.csr_cuda_dirty[i] = True
            if int(self.online_edges_total) != before:
                for i in range(len(state.overlay_dirty)):
                    state.overlay_dirty[i] = True

    def _get_merged_csr(self, state, kv_hdx: int, kv_len: int):
        kv_hdx = int(kv_hdx)
        kv_len = int(kv_len)
        if (
            kv_hdx < len(state.csr_offsets)
            and not state.csr_dirty[kv_hdx]
            and int(state.csr_len[kv_hdx]) == kv_len
        ):
            return state.csr_offsets[kv_hdx], state.csr_neighbors[kv_hdx]

        base_offsets = state.base_offsets[kv_hdx] if kv_hdx < len(state.base_offsets) else np.zeros((1,), dtype=np.uint32)
        base_neighbors = state.base_neighbors[kv_hdx] if kv_hdx < len(state.base_neighbors) else np.empty((0,), dtype=np.int32)
        extra = state.extra_edges[kv_hdx] if kv_hdx < len(state.extra_edges) else {}
        rows = []
        total = 0
        offsets = np.zeros((kv_len + 1,), dtype=np.uint32)
        base_rows = max(0, int(base_offsets.shape[0]) - 1)
        for node in range(kv_len):
            row = []
            if node < base_rows:
                start = int(base_offsets[node])
                end = int(base_offsets[node + 1])
                row.extend(int(x) for x in base_neighbors[start:end])
            if node in extra:
                row.extend(int(x) for x in extra.get(node, ()))
            if row:
                row = sorted({x for x in row if 0 <= int(x) < kv_len and int(x) != node})
            rows.append(row)
            total += len(row)
            offsets[node + 1] = np.uint32(total)
        neighbors = np.empty((total,), dtype=np.int32)
        pos = 0
        for row in rows:
            if row:
                neighbors[pos:pos + len(row)] = np.asarray(row, dtype=np.int32)
                pos += len(row)
        state.csr_offsets[kv_hdx] = np.ascontiguousarray(offsets, dtype=np.uint32)
        state.csr_neighbors[kv_hdx] = np.ascontiguousarray(neighbors, dtype=np.int32)
        state.csr_len[kv_hdx] = kv_len
        state.csr_dirty[kv_hdx] = False
        if kv_hdx < len(state.csr_cuda_dirty):
            state.csr_cuda_dirty[kv_hdx] = True
        return state.csr_offsets[kv_hdx], state.csr_neighbors[kv_hdx]

    def _get_merged_csr_cuda(self, state, kv_hdx: int, kv_len: int, device):
        offsets, neighbors = self._get_merged_csr(state, kv_hdx, kv_len)
        kv_hdx = int(kv_hdx)
        while kv_hdx >= len(state.csr_offsets_cuda):
            state.csr_offsets_cuda.append(None)
            state.csr_neighbors_cuda.append(None)
            state.csr_cuda_len.append(-1)
            state.csr_cuda_device.append("")
            state.csr_cuda_dirty.append(True)

        device_key = str(device)
        if (
            not state.csr_cuda_dirty[kv_hdx]
            and int(state.csr_cuda_len[kv_hdx]) == int(kv_len)
            and state.csr_cuda_device[kv_hdx] == device_key
            and state.csr_offsets_cuda[kv_hdx] is not None
            and state.csr_neighbors_cuda[kv_hdx] is not None
        ):
            return state.csr_offsets_cuda[kv_hdx], state.csr_neighbors_cuda[kv_hdx]

        state.csr_offsets_cuda[kv_hdx] = torch.as_tensor(
            offsets.astype(np.int64, copy=False),
            dtype=torch.int64,
            device=device,
        )
        state.csr_neighbors_cuda[kv_hdx] = torch.as_tensor(
            neighbors.astype(np.int32, copy=False),
            dtype=torch.int32,
            device=device,
        )
        state.csr_cuda_len[kv_hdx] = int(kv_len)
        state.csr_cuda_device[kv_hdx] = device_key
        state.csr_cuda_dirty[kv_hdx] = False
        self.csr_cuda_uploads += 1
        return state.csr_offsets_cuda[kv_hdx], state.csr_neighbors_cuda[kv_hdx]

    @staticmethod
    def _next_capacity(n: int):
        n = max(1, int(n))
        return 1 << (n - 1).bit_length()

    def _get_base_csr_cuda_fullgpu(self, state, kv_hdx: int, kv_len: int, device):
        kv_hdx = int(kv_hdx)
        kv_len = int(kv_len)
        while kv_hdx >= len(state.base_offsets_cuda):
            state.base_offsets_cuda.append(None)
            state.base_neighbors_cuda.append(None)
            state.base_csr_cuda_capacity.append(0)
            state.base_csr_cuda_len.append(0)
            state.base_csr_cuda_device.append("")

        base_offsets = state.base_offsets[kv_hdx] if kv_hdx < len(state.base_offsets) else np.zeros((1,), dtype=np.uint32)
        base_neighbors = state.base_neighbors[kv_hdx] if kv_hdx < len(state.base_neighbors) else np.empty((0,), dtype=np.int32)
        edge_count = int(base_offsets[-1]) if int(base_offsets.shape[0]) > 0 else 0
        needed = kv_len + 1
        device_key = str(device)
        cached_offsets = state.base_offsets_cuda[kv_hdx]
        cached_neighbors = state.base_neighbors_cuda[kv_hdx]
        cached_capacity = int(state.base_csr_cuda_capacity[kv_hdx])
        cached_len = int(state.base_csr_cuda_len[kv_hdx])
        cache_ok = (
            cached_offsets is not None
            and cached_neighbors is not None
            and cached_capacity >= needed
            and state.base_csr_cuda_device[kv_hdx] == device_key
        )
        if not cache_ok:
            capacity = self._next_capacity(needed)
            offsets_t = torch.empty((capacity,), dtype=torch.int64, device=device)
            offsets_t.fill_(edge_count)
            prefix_len = min(int(base_offsets.shape[0]), capacity)
            if prefix_len > 0:
                offsets_t[:prefix_len] = torch.as_tensor(
                    base_offsets[:prefix_len].astype(np.int64, copy=False),
                    dtype=torch.int64,
                    device=device,
                )
            neighbors_t = torch.as_tensor(
                base_neighbors.astype(np.int32, copy=False),
                dtype=torch.int32,
                device=device,
            )
            state.base_offsets_cuda[kv_hdx] = offsets_t
            state.base_neighbors_cuda[kv_hdx] = neighbors_t
            state.base_csr_cuda_capacity[kv_hdx] = int(capacity)
            state.base_csr_cuda_device[kv_hdx] = device_key
            state.base_csr_cuda_len[kv_hdx] = int(needed)
            self.base_csr_cuda_uploads += 1
            return offsets_t[:needed], neighbors_t
        else:
            offsets_t = cached_offsets
            neighbors_t = cached_neighbors

        if cached_len < needed:
            offsets_t[cached_len:needed].fill_(edge_count)
            prefix_start = max(0, cached_len)
            prefix_end = min(int(base_offsets.shape[0]), needed)
            if prefix_end > prefix_start:
                offsets_t[prefix_start:prefix_end] = torch.as_tensor(
                    base_offsets[prefix_start:prefix_end].astype(np.int64, copy=False),
                    dtype=torch.int64,
                    device=device,
                )
        state.base_csr_cuda_len[kv_hdx] = max(cached_len, needed)
        return offsets_t[:needed], neighbors_t

    def _get_overlay_cuda_fullgpu(self, state, kv_hdx: int, kv_len: int, device):
        kv_hdx = int(kv_hdx)
        kv_len = int(kv_len)
        while kv_hdx >= len(state.overlay_counts_cuda):
            state.overlay_counts_cuda.append(None)
            state.overlay_neighbors_cuda.append(None)
            state.overlay_cuda_capacity.append(0)
            state.overlay_cuda_width.append(0)
            state.overlay_cuda_device.append("")
            state.overlay_dirty.append(True)

        extra = state.extra_edges[kv_hdx] if kv_hdx < len(state.extra_edges) else {}
        max_row = 0
        for node, vals in extra.items():
            if 0 <= int(node) < kv_len:
                max_row = max(max_row, len(vals))
        width = max(1, min(64, max_row))
        needed = kv_len
        device_key = str(device)
        cache_ok = (
            not state.overlay_dirty[kv_hdx]
            and state.overlay_counts_cuda[kv_hdx] is not None
            and state.overlay_neighbors_cuda[kv_hdx] is not None
            and int(state.overlay_cuda_capacity[kv_hdx]) >= needed
            and int(state.overlay_cuda_width[kv_hdx]) == width
            and state.overlay_cuda_device[kv_hdx] == device_key
        )
        if cache_ok:
            return (
                state.overlay_counts_cuda[kv_hdx][:needed],
                state.overlay_neighbors_cuda[kv_hdx][:needed, :width],
            )

        capacity = self._next_capacity(needed)
        counts = torch.zeros((capacity,), dtype=torch.int32, device=device)
        neighbors = torch.full((capacity, width), -1, dtype=torch.int32, device=device)
        if extra:
            counts_cpu = np.zeros((needed,), dtype=np.int32)
            neighbors_cpu = np.full((needed, width), -1, dtype=np.int32)
            for node, vals in extra.items():
                node = int(node)
                if node < 0 or node >= needed:
                    continue
                row = sorted({int(x) for x in vals if 0 <= int(x) < needed and int(x) != node})
                if not row:
                    continue
                row = row[:width]
                counts_cpu[node] = int(len(row))
                neighbors_cpu[node, :len(row)] = np.asarray(row, dtype=np.int32)
            counts[:needed] = torch.as_tensor(counts_cpu, dtype=torch.int32, device=device)
            neighbors[:needed, :width] = torch.as_tensor(neighbors_cpu, dtype=torch.int32, device=device)

        state.overlay_counts_cuda[kv_hdx] = counts
        state.overlay_neighbors_cuda[kv_hdx] = neighbors
        state.overlay_cuda_capacity[kv_hdx] = int(capacity)
        state.overlay_cuda_width[kv_hdx] = int(width)
        state.overlay_cuda_device[kv_hdx] = device_key
        state.overlay_dirty[kv_hdx] = False
        self.overlay_cuda_uploads += 1
        return counts[:needed], neighbors[:needed, :width]

    def _select_from_cpp(self, state, hdx: int, kv_hdx: int, kv_len: int, dyn_start: int, dyn_end: int, seeds, init_scores):
        offsets, neighbors = self._get_merged_csr(state, kv_hdx, kv_len)
        query = state.current_query_states[0, hdx, 0, :].detach().float().cpu().numpy()
        keys = state.keys_cpu[kv_hdx]
        ids, _scores, meta = self.roar.search_roar_graph_csr_cpp(
            query=query,
            keys=keys,
            offsets=offsets,
            neighbors=neighbors,
            init_ids=np.asarray(seeds, dtype=np.int32),
            init_scores=np.asarray(init_scores, dtype=np.float32),
            topk=int(self.topk),
            lpq=int(self.candidate_target),
            max_cmps=int(self.visit_budget),
            max_hops=int(self.visit_budget),
            dynamic_start=int(dyn_start),
            dynamic_end=int(dyn_end),
            num_threads=int(self.roar_threads),
            score_agg="max",
            key_dtype="fp32",
        )
        self.visited_total += int(meta.get("visited", 0)) if isinstance(meta, dict) else 0
        self.candidates_total += int(meta.get("queue_size", len(ids))) if isinstance(meta, dict) else int(len(ids))
        if isinstance(meta, dict):
            self.stop_reasons[str(meta.get("stop_reason", "unknown"))] += 1
        return [int(x) for x in np.asarray(ids, dtype=np.int32).tolist()[: int(self.topk)]]

    def _select_group_cuda(self, state, kv_hdx: int, head_ids, kv_len: int, dyn_start: int, dyn_end: int, seeds_by_head, scores_by_head):
        if not torch.cuda.is_available() or not self.roar.roargraph_cuda_available():
            self.cuda_fallbacks += len(head_ids)
            return {
                hdx: self._select_from_cpp(state, hdx, kv_hdx, kv_len, dyn_start, dyn_end, seeds_by_head[hdx], scores_by_head[hdx])
                for hdx in head_ids
            }
        keys_t = state.keys_cuda[kv_hdx]
        if keys_t is None or keys_t.device.type != "cuda":
            self.cuda_fallbacks += len(head_ids)
            return {
                hdx: self._select_from_cpp(state, hdx, kv_hdx, kv_len, dyn_start, dyn_end, seeds_by_head[hdx], scores_by_head[hdx])
                for hdx in head_ids
            }

        offsets, neighbors = self._get_merged_csr(state, kv_hdx, kv_len)
        device = keys_t.device
        queries = torch.stack(
            [state.current_query_states[0, hdx, 0, :].detach().float().to(device) for hdx in head_ids],
            dim=0,
        ).contiguous()
        init_width = max(1, max((len(seeds_by_head[hdx]) for hdx in head_ids), default=0))
        init_ids = torch.full((len(head_ids), init_width), -1, dtype=torch.int32)
        init_scores = torch.full((len(head_ids), init_width), float("-inf"), dtype=torch.float32)
        for row, hdx in enumerate(head_ids):
            seeds = list(seeds_by_head[hdx])
            vals = list(scores_by_head[hdx])
            if seeds:
                init_ids[row, :len(seeds)] = torch.as_tensor(seeds, dtype=torch.int32)
                init_scores[row, :len(vals)] = torch.as_tensor(vals, dtype=torch.float32)
        offsets_t = torch.as_tensor(offsets.astype(np.int64, copy=False), dtype=torch.int64)
        neighbors_t = torch.as_tensor(neighbors.astype(np.int32, copy=False), dtype=torch.int32)
        ids_t, _scores_t, counts_t, visited_t, stop_t = self.roar.search_roar_graph_csr_cuda_group(
            queries_seed=queries,
            queries_rank=queries,
            keys=keys_t.float().contiguous(),
            offsets=offsets_t,
            neighbors=neighbors_t,
            init_ids=init_ids,
            init_scores=init_scores,
            token_budget=int(self.topk),
            candidate_target=int(self.candidate_target),
            expand_width=int(self.expand_width),
            min_visits=int(self.min_visits),
            max_visits=int(self.visit_budget),
            frontier_topn=int(self.frontier_topn),
            stop_patience=int(self.stop_patience),
            stop_margin=float(self.stop_margin),
            dynamic_start=int(dyn_start),
            dynamic_end=int(dyn_end),
            score_agg="max",
        )
        ids_t = ids_t.cpu()
        counts_t = counts_t.cpu()
        visited_t = visited_t.cpu()
        stop_t = stop_t.cpu()
        out = {}
        stop_map = {
            0: "frontier_empty",
            1: "max_visits",
            2: "candidate_cap",
            3: "stability_gap",
            4: "empty_init",
        }
        for row, hdx in enumerate(head_ids):
            count = max(0, min(int(counts_t[row].item()), int(self.topk)))
            selected = [int(x) for x in ids_t[row, :count].tolist() if int(x) >= 0]
            out[int(hdx)] = selected
            self.visited_total += int(visited_t[row].item())
            self.candidates_total += int(counts_t[row].item())
            self.stop_reasons[stop_map.get(int(stop_t[row].item()), f"code_{int(stop_t[row].item())}")] += 1
        return out

    def _select_group_fullgpu(self, state, kv_hdx: int, head_ids, kv_len: int, dyn_start: int, dyn_end: int, seeds_by_head, scores_by_head):
        if not torch.cuda.is_available() or not self.roar.roargraph_cuda_kernel_available():
            self.cuda_fallbacks += len(head_ids)
            return self._select_group_cuda(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)
        keys_t = state.keys_cuda[kv_hdx]
        if keys_t is None or keys_t.device.type != "cuda":
            self.cuda_fallbacks += len(head_ids)
            return self._select_group_cuda(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)

        device = keys_t.device
        queries = torch.stack(
            [state.current_query_states[0, hdx, 0, :].detach().float().to(device) for hdx in head_ids],
            dim=0,
        ).contiguous()
        if int(queries.shape[-1]) > 256:
            self.cuda_fallbacks += len(head_ids)
            self.fullgpu_fallback_reasons["head_dim_gt_256"] += len(head_ids)
            return self._select_group_cuda(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)
        offsets_t, neighbors_t = self._get_base_csr_cuda_fullgpu(state, kv_hdx, kv_len, device)
        overlay_counts_t, overlay_neighbors_t = self._get_overlay_cuda_fullgpu(state, kv_hdx, kv_len, device)

        prev_width = self._ensure_fullgpu_seed_tensors(
            state,
            module=None,
            device=device,
            kv_heads=len(state.hubs),
            heads=len(state.prev_seeds),
        )
        head_indices_t = torch.as_tensor([int(hdx) for hdx in head_ids], dtype=torch.long, device=device)
        prev_seed_ids = torch.index_select(state.prev_seed_ids_cuda, 0, head_indices_t)
        prev_seed_counts = torch.index_select(state.prev_seed_counts_cuda, 0, head_indices_t)
        hub_seed_ids = state.hub_seed_ids_cuda[int(kv_hdx)] if int(kv_hdx) < len(state.hub_seed_ids_cuda) else torch.empty((0,), dtype=torch.int32, device=device)
        q_count = len(head_ids)
        zeros = torch.zeros((q_count,), dtype=torch.float32, device=device)
        try:
            ids_t, _scores_t, counts_t, visited_t, stop_t, next_prev_ids_t, next_prev_counts_t, debug_t, _keep_t, _mass_t = (
                self.roar.search_roar_graph_csr_cuda_group_fullgpu(
                queries_seed=queries,
                queries_rank=queries,
                queries_attn=queries,
                keys=keys_t,
                attn_keys=keys_t,
                static_logz=zeros,
                upper_scores=zeros,
                total_score_sum=zeros,
                total_score_sumsq=zeros,
                offsets=offsets_t,
                neighbors=neighbors_t,
                overlay_counts=overlay_counts_t,
                overlay_neighbors=overlay_neighbors_t,
                prev_seed_ids=prev_seed_ids,
                prev_seed_counts=prev_seed_counts,
                hub_seed_ids=hub_seed_ids,
                token_budget=int(self.topk),
                candidate_target=int(self.candidate_target),
                beam_width=min(64, int(self.expand_width)),
                max_degree=min(16, max(1, int(self.graph_degree))),
                min_visits=int(self.min_visits),
                max_visits=int(self.visit_budget),
                stop_patience=int(self.stop_patience),
                stop_margin=float(self.stop_margin),
                dynamic_start=int(dyn_start),
                dynamic_end=int(dyn_end),
                seed_k=max(1, int(self.candidate_target)),
                seed_floor=0,
                seed_tail_k=max(0, int(self.seed_count)),
                seed_prev_k=int(prev_width),
                adaptive_enable=False,
                adaptive_min_keep=1,
                adaptive_target_omass=0.0,
                adaptive_prior_mode="global_norm",
                adaptive_prior_var_scale=1.0,
                score_agg="max",
                )
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "roar_cuda_fullgpu" not in msg:
                raise
            self.cuda_fallbacks += len(head_ids)
            if "head_dim" in msg:
                reason = "head_dim_contract"
            elif "max_degree" in msg:
                reason = "max_degree_contract"
            else:
                reason = "runtime_contract"
            self.fullgpu_fallback_reasons[reason] += len(head_ids)
            return self._select_group_cuda(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)

        ids_cpu = ids_t.detach().cpu()
        counts_cpu = counts_t.detach().cpu()
        visited_cpu = visited_t.detach().cpu()
        stop_cpu = stop_t.detach().cpu()
        debug_cpu = debug_t.detach().cpu()
        state.prev_seed_ids_cuda.index_copy_(0, head_indices_t, next_prev_ids_t)
        state.prev_seed_counts_cuda.index_copy_(0, head_indices_t, next_prev_counts_t)
        out = {}
        stop_map = {
            0: "frontier_empty",
            1: "max_visits",
            2: "candidate_cap",
            3: "stability_gap",
            4: "empty_init",
            5: "adaptive_bound",
            6: "adaptive_exact_cover",
        }
        for row, hdx in enumerate(head_ids):
            count = max(0, min(int(counts_cpu[row].item()), int(self.topk)))
            selected = [int(x) for x in ids_cpu[row, :count].tolist() if int(x) >= 0]
            out[int(hdx)] = selected
            self.visited_total += int(visited_cpu[row].item())
            self.candidates_total += int(debug_cpu[row, 0].item()) if debug_cpu.ndim == 2 and debug_cpu.shape[1] > 0 else int(counts_cpu[row].item())
            self.stop_reasons[stop_map.get(int(stop_cpu[row].item()), f"code_{int(stop_cpu[row].item())}")] += 1
        return out

    def _graph_keep_mask(self, state, scores, module):
        batch, heads, q_len, kv_len = scores.shape
        if batch != 1 or q_len != 1:
            raise RuntimeError("graph_topk_roar attention currently supports batch=1 and decode q_len=1 only.")
        dyn_start, dyn_end = self._dynamic_bounds(kv_len)
        keep = torch.zeros((batch, heads, kv_len), dtype=torch.bool, device=scores.device)
        if dyn_start > 0:
            keep[:, :, :dyn_start] = True
        if dyn_end < kv_len:
            keep[:, :, dyn_end:] = True

        selected_by_kv = [[] for _ in range(module_num_key_value_heads(module))]
        grouped_heads = collections.defaultdict(list)
        seeds_by_head = {}
        scores_by_head = {}
        if self.search_backend == "cuda_fullgpu":
            if scores.device.type == "cuda":
                self._ensure_fullgpu_seed_tensors(state, module, scores.device)
            for hdx in range(int(heads)):
                kv_hdx = int(hdx) // int(module.num_key_value_groups)
                grouped_heads[kv_hdx].append(hdx)
        else:
            for hdx in range(int(heads)):
                kv_hdx = int(hdx) // int(module.num_key_value_groups)
                seeds = self._seed_nodes(state, hdx, kv_hdx, kv_len, dyn_start, dyn_end)
                if not seeds:
                    continue
                init_scores = [float(x) for x in scores[0, hdx, 0, torch.as_tensor(seeds, device=scores.device)].detach().float().cpu().tolist()]
                seeds_by_head[hdx] = seeds
                scores_by_head[hdx] = init_scores
                grouped_heads[kv_hdx].append(hdx)

        for kv_hdx, head_ids in grouped_heads.items():
            if self.search_backend == "cuda_fullgpu":
                selected_map = self._select_group_fullgpu(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)
            elif self.search_backend == "cuda_group":
                selected_map = self._select_group_cuda(state, kv_hdx, head_ids, kv_len, dyn_start, dyn_end, seeds_by_head, scores_by_head)
            else:
                selected_map = {
                    hdx: self._select_from_cpp(state, hdx, kv_hdx, kv_len, dyn_start, dyn_end, seeds_by_head[hdx], scores_by_head[hdx])
                    for hdx in head_ids
                }
            for hdx, selected in selected_map.items():
                if not selected:
                    continue
                idx = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
                keep[0, int(hdx)].scatter_(dim=-1, index=idx, value=True)
                if self.search_backend != "cuda_fullgpu":
                    state.prev_seeds[int(hdx)] = selected[: self.seed_count]
                selected_by_kv[int(kv_hdx)].extend(selected[: self.online_edges])

        for kv_hdx, selected in enumerate(selected_by_kv):
            if not selected:
                continue
            deduped = []
            seen = set()
            for token_idx in selected:
                token_idx = int(token_idx)
                if token_idx in seen:
                    continue
                deduped.append(token_idx)
                seen.add(token_idx)
                if len(deduped) >= self.online_edges:
                    break
            state.pending_edges[kv_hdx] = deduped

        if not bool(keep.any()):
            keep[:, :, -1:] = True
        return keep

    def summary(self):
        summary = super().summary()
        summary["mode"] = "graph_topk_roar"
        summary["search_backend"] = str(self.search_backend)
        summary["candidate_target"] = int(self.candidate_target)
        summary["expand_width"] = int(self.expand_width)
        summary["min_visits"] = int(self.min_visits)
        summary["frontier_topn"] = int(self.frontier_topn)
        summary["stop_patience"] = int(self.stop_patience)
        summary["stop_margin"] = float(self.stop_margin)
        summary["roar_cand_limit"] = int(self.roar_cand_limit)
        summary["roar_enhance_limit"] = int(self.roar_enhance_limit)
        summary["roar_entry"] = str(self.roar_entry)
        summary["roar_threads"] = int(self.roar_threads)
        summary["cuda_fallbacks"] = int(self.cuda_fallbacks)
        summary["fullgpu_fallback_reasons"] = dict(self.fullgpu_fallback_reasons)
        summary["csr_cuda_uploads"] = int(self.csr_cuda_uploads)
        summary["base_csr_cuda_uploads"] = int(self.base_csr_cuda_uploads)
        summary["overlay_cuda_uploads"] = int(self.overlay_cuda_uploads)
        summary["key_cuda_uploads"] = int(self.key_cuda_uploads)
        summary["key_cuda_append_updates"] = int(self.key_cuda_append_updates)
        summary["prev_seed_cuda_uploads"] = int(self.prev_seed_cuda_uploads)
        summary["stop_reasons"] = dict(self.stop_reasons)
        return summary


def maybe_install_hf_attention_backend(model, inventory, args):
    if args.hf_attention_mode == "native":
        return None
    if args.hf_attention_mode == "oracle_topk":
        return HFOracleTopKPatcher(
            model=model,
            inventory=inventory,
            topk=int(args.hf_sparse_topk),
            static_prefix=int(args.hf_sparse_static_prefix),
            static_suffix=int(args.hf_sparse_static_suffix),
        ).install()
    if args.hf_attention_mode == "graph_topk":
        return HFGraphTopKPatcher(
            model=model,
            inventory=inventory,
            topk=int(args.hf_sparse_topk),
            static_prefix=int(args.hf_sparse_static_prefix),
            static_suffix=int(args.hf_sparse_static_suffix),
            graph_degree=int(args.hf_graph_degree),
            visit_budget=int(args.hf_graph_visit_budget),
            seed_count=int(args.hf_graph_seed_count),
            online_edges=int(args.hf_graph_online_edges),
        ).install()
    if args.hf_attention_mode == "graph_topk_roar":
        return HFGraphTopKRoarPatcher(
            model=model,
            inventory=inventory,
            topk=int(args.hf_sparse_topk),
            static_prefix=int(args.hf_sparse_static_prefix),
            static_suffix=int(args.hf_sparse_static_suffix),
            graph_degree=int(args.hf_graph_degree),
            visit_budget=int(args.hf_graph_visit_budget),
            seed_count=int(args.hf_graph_seed_count),
            online_edges=int(args.hf_graph_online_edges),
            search_backend=str(args.hf_graph_search_backend),
            candidate_target=int(args.hf_graph_candidate_target),
            expand_width=int(args.hf_graph_expand_width),
            min_visits=int(args.hf_graph_min_visits),
            frontier_topn=int(args.hf_graph_frontier_topn),
            stop_patience=int(args.hf_graph_stop_patience),
            stop_margin=float(args.hf_graph_stop_margin),
            roar_cand_limit=int(args.hf_graph_roar_cand_limit),
            roar_enhance_limit=int(args.hf_graph_roar_enhance_limit),
            roar_entry=str(args.hf_graph_roar_entry),
            roar_threads=int(args.hf_graph_roar_threads),
        ).install()
    raise ValueError(f"Unsupported HF attention mode: {args.hf_attention_mode}")


class AttentionTrace:
    def __init__(self, model, config, max_records: int):
        self.model = model
        self.config = config
        self.max_records = int(max_records)
        self.records = []
        self.handles = []
        self.enabled = False

    @staticmethod
    def _shape(value):
        if isinstance(value, torch.Tensor):
            return list(value.shape)
        if isinstance(value, (list, tuple)):
            return [AttentionTrace._shape(v) for v in value[:4]]
        if isinstance(value, dict):
            return {str(k): AttentionTrace._shape(v) for k, v in list(value.items())[:8]}
        return type(value).__name__

    def _hook(self, name, module):
        def fn(_module, inputs, output):
            if not self.enabled or len(self.records) >= self.max_records:
                return
            self.records.append(
                {
                    "name": name,
                    "class": type(module).__name__,
                    "input_shapes": self._shape(inputs),
                    "output_shapes": self._shape(output),
                }
            )
        return fn

    def install(self):
        layer_types = config_get(get_text_config(self.config), "layer_types", [])
        for name, module in self.model.named_modules():
            if classify_attention_module(name, module, layer_types) is not None:
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_hf_model(args, dtype, config):
    ensure_psutil_virtual_memory()
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if config is not None:
        kwargs["config"] = config

    attempts = []

    def add_transformers_class(class_name: str):
        try:
            import transformers

            cls = getattr(transformers, class_name)
            attempts.append((class_name, cls))
        except Exception as exc:
            errors.append(f"{class_name}: unavailable: {type(exc).__name__}: {exc}")

    errors = []
    text_model_type = str(config_get(get_text_config(config), "model_type", "") or "").lower()
    top_model_type = str(config_get(config, "model_type", "") or "").lower()
    architectures = config_get(config, "architectures", []) or []
    text_architectures = config_get(get_text_config(config), "architectures", []) or []
    if bool(args.hf_language_model_only):
        if text_model_type in {"qwen3", "qwen3_5", "qwen3_5_text"} or top_model_type in {"qwen3", "qwen3_5"}:
            for class_name in (
                "Qwen3_5ForCausalLM",
                "Qwen3ForCausalLM",
                "AutoModelForCausalLM",
            ):
                add_transformers_class(class_name)
    for class_name in list(architectures) + list(text_architectures):
        add_transformers_class(str(class_name))

    # Current project venv has a narrow Transformers install where AutoModelForCausalLM can be broken.
    # Keep Auto attempts for newer model envs, but also add concrete classes for cached baselines.
    if text_model_type == "llama" or top_model_type == "llama":
        add_transformers_class("LlamaForCausalLM")
    if text_model_type in {"qwen2", "qwen2_moe"} or top_model_type in {"qwen2", "qwen2_moe"}:
        add_transformers_class("Qwen2ForCausalLM")
    if text_model_type in {"qwen3", "qwen3_5"} or top_model_type in {"qwen3", "qwen3_5"}:
        for class_name in (
            "Qwen3ForCausalLM",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
        ):
            add_transformers_class(class_name)
    if text_model_type == "mistral" or top_model_type == "mistral":
        add_transformers_class("MistralForCausalLM")
    if text_model_type in {"mistral3"} or top_model_type in {"mistral3"}:
        for class_name in (
            "Mistral3ForConditionalGeneration",
            "Mistral3ForCausalLM",
        ):
            add_transformers_class(class_name)
    if text_model_type in {"glm", "chatglm"} or top_model_type in {"glm", "chatglm"}:
        for class_name in (
            "GlmForCausalLM",
            "GLMForCausalLM",
            "ChatGLMForConditionalGeneration",
        ):
            add_transformers_class(class_name)
    for class_name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        add_transformers_class(class_name)

    deduped = []
    seen = set()
    for label, cls in attempts:
        if label in seen:
            continue
        seen.add(label)
        deduped.append((label, cls))
    attempts = deduped

    for label, cls in attempts:
        try:
            model = cls.from_pretrained(args.model_name, **kwargs)
            model.eval()
            return model, label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Failed to load model with available HF auto classes:\n" + "\n".join(errors))


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def summarize_tokenizer(tokenizer):
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "padding_side": tokenizer.padding_side,
    }


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_with_optional_mask(model, input_ids, past_key_values=None, attention_mask=None):
    kwargs = {"input_ids": input_ids, "use_cache": True}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("attention_mask", None)
        return model(**kwargs)


def append_ones_attention_mask(attention_mask, n_tokens: int, device):
    extra = torch.ones((attention_mask.shape[0], int(n_tokens)), dtype=attention_mask.dtype, device=device)
    return torch.cat([attention_mask, extra], dim=-1)


def build_codebook_token_map(tokenizer):
    token_map = []
    seen = set()
    for word in CODEBOOK:
        ids = tokenizer(" " + word, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if len(ids) != 1:
            ids = tokenizer(word, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if len(ids) != 1:
            raise RuntimeError(f"Codebook word {word!r} is not single-token under the current tokenizer.")
        tok_id = int(ids[0])
        if tok_id in seen:
            raise RuntimeError(f"Codebook token collision for id={tok_id} on word {word!r}.")
        seen.add(tok_id)
        token_map.append((tok_id, word))
    return token_map


def append_text_to_cache(model, tokenizer, past, attention_mask, logits, text: str, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if int(ids.numel()) == 0:
        return past, attention_mask, logits, ids
    attention_mask = append_ones_attention_mask(attention_mask, int(ids.shape[1]), device)
    out = forward_with_optional_mask(
        model,
        input_ids=ids,
        past_key_values=past,
        attention_mask=attention_mask,
    )
    return out.past_key_values, attention_mask, out.logits[:, -1:, :], ids


def sample_next_id(logits):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def decode_one_token(model, token_id, past, attention_mask, device):
    attention_mask = append_ones_attention_mask(attention_mask, 1, device)
    out = forward_with_optional_mask(
        model,
        input_ids=token_id,
        past_key_values=past,
        attention_mask=attention_mask,
    )
    return out.past_key_values, attention_mask, out.logits[:, -1:, :]


def decode_answers_with_prefix_scaffold(
    model,
    tokenizer,
    past,
    attention_mask,
    logits,
    expected_answers: int,
    max_new_tokens: int,
    constrained_token_map,
):
    device = model_device(model)
    rendered_text = ""
    generated_ids = []
    remaining = int(max(1, max_new_tokens))
    allowed_ids_t = None
    allowed_words = None
    if constrained_token_map:
        allowed_ids_t = torch.as_tensor(
            [int(tok_id) for tok_id, _word in constrained_token_map],
            dtype=torch.long,
            device=device,
        )
        allowed_words = [str(word) for _tok_id, word in constrained_token_map]

    for answer_idx in range(int(expected_answers)):
        prefix = f"ANSWER {answer_idx + 1}: "
        if rendered_text and not rendered_text.endswith("\n"):
            prefix = "\n" + prefix
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prefix_token_count = int(prefix_ids.shape[0])
        if remaining < prefix_token_count:
            break
        rendered_text += prefix
        past, attention_mask, logits, _ = append_text_to_cache(
            model=model,
            tokenizer=tokenizer,
            past=past,
            attention_mask=attention_mask,
            logits=logits,
            text=prefix,
            device=device,
        )
        remaining -= prefix_token_count
        if remaining <= 0:
            break

        if allowed_ids_t is not None:
            logits_view = logits[:, -1, :] if logits.dim() == 3 else logits
            allowed_logits = torch.index_select(logits_view, dim=-1, index=allowed_ids_t)
            best_idx = int(torch.argmax(allowed_logits, dim=-1).item())
            chosen_id = int(allowed_ids_t[best_idx].item())
            chosen_word = allowed_words[best_idx]
            token_id = torch.tensor([[chosen_id]], dtype=torch.long, device=device)
            generated_ids.append(chosen_id)
            rendered_text += chosen_word
            if answer_idx + 1 < int(expected_answers):
                rendered_text += "\n"
            past, attention_mask, logits = decode_one_token(
                model=model,
                token_id=token_id,
                past=past,
                attention_mask=attention_mask,
                device=device,
            )
            remaining -= 1
            continue

        token_id = sample_next_id(logits)
        value_ids = []
        for _ in range(remaining):
            value_id = int(token_id[0, 0].item())
            generated_ids.append(value_id)
            value_ids.append(value_id)
            value_text = tokenizer.decode(value_ids, skip_special_tokens=True)
            past, attention_mask, logits = decode_one_token(
                model=model,
                token_id=token_id,
                past=past,
                attention_mask=attention_mask,
                device=device,
            )
            remaining -= 1
            if "\n" in value_text:
                rendered_text += value_text.split("\n", 1)[0] + "\n"
                break
            if remaining <= 0:
                rendered_text += value_text
                break
            token_id = sample_next_id(logits)

    parsed_answers = len(list(ANSWER_RE.finditer(rendered_text)))
    return rendered_text, generated_ids, past, attention_mask, logits, bool(parsed_answers >= int(expected_answers))


@torch.inference_mode()
def hf_manual_cache_run(model, tokenizer, ledger_prompt: str, question_prompt: str, max_ledger_tokens: int, max_answer_tokens: int, args, trace=None):
    device = model_device(model)
    prompt_text = maybe_apply_chat_template(tokenizer, ledger_prompt, args)
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else torch.ones_like(input_ids)

    if trace is not None:
        trace.enabled = bool(args.trace_prefill)
    prefill_start = time.time()
    out = forward_with_optional_mask(model, input_ids=input_ids, attention_mask=attention_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_sec = float(time.time() - prefill_start)
    past = out.past_key_values
    logits = out.logits[:, -1:, :]

    ledger_ids = []
    if trace is not None:
        trace.enabled = True
    decode_start = time.time()
    for step in range(int(max_ledger_tokens)):
        next_id = sample_next_id(logits)
        ledger_ids.append(int(next_id[0, 0].item()))
        if trace is not None and step >= int(args.trace_decode_steps):
            trace.enabled = False
        past, attention_mask, logits = decode_one_token(
            model,
            token_id=next_id,
            past=past,
            attention_mask=attention_mask,
            device=device,
        )
        ledger_text = tokenizer.decode(ledger_ids, skip_special_tokens=True)
        if not args.force_max_decode_steps and "END LEDGER" in ledger_text:
            break

    question_text = maybe_apply_chat_template(tokenizer, question_prompt, args) if args.use_chat_template else question_prompt
    question_ids = tokenizer(question_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if int(question_ids.numel()) > 0:
        attention_mask = append_ones_attention_mask(attention_mask, int(question_ids.shape[1]), device)
        out = forward_with_optional_mask(
            model,
            input_ids=question_ids,
            past_key_values=past,
            attention_mask=attention_mask,
        )
        past = out.past_key_values
        logits = out.logits[:, -1:, :]

    trace_answer_start = len(trace.records) if trace is not None else 0
    if args.answer_prefix_scaffold:
        constrained_token_map = build_codebook_token_map(tokenizer) if args.answer_constrained_codebook else None
        answer_text, answer_ids, past, attention_mask, logits, _answer_done = decode_answers_with_prefix_scaffold(
            model=model,
            tokenizer=tokenizer,
            past=past,
            attention_mask=attention_mask,
            logits=logits,
            expected_answers=int(args.num_queries),
            max_new_tokens=int(max_answer_tokens),
            constrained_token_map=constrained_token_map,
        )
    else:
        answer_ids = []
        for _step in range(int(max_answer_tokens)):
            next_id = sample_next_id(logits)
            answer_ids.append(int(next_id[0, 0].item()))
            past, attention_mask, logits = decode_one_token(
                model,
                token_id=next_id,
                past=past,
                attention_mask=attention_mask,
                device=device,
            )
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            if not args.force_max_decode_steps and len(list(ANSWER_RE.finditer(answer_text))) >= int(args.num_queries):
                break
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_sec = float(time.time() - decode_start)
    if trace is not None:
        trace.enabled = False

    ledger_text = tokenizer.decode(ledger_ids, skip_special_tokens=True)
    output_text = ledger_prompt + ledger_text + question_prompt + answer_text
    return {
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "prompt_tokens": int(input_ids.shape[1]),
        "ledger_generated_tokens": int(len(ledger_ids)),
        "question_tokens": int(question_ids.shape[1]),
        "answer_generated_tokens": int(len(answer_ids)),
        "ledger_output": ledger_text,
        "answer_output": answer_text,
        "output": output_text,
        "force_max_decode_steps": bool(args.force_max_decode_steps),
        "trace_answer_start_record": int(trace_answer_start),
    }


@torch.inference_mode()
def hf_generate_reprefill_run(model, tokenizer, ledger_prompt: str, question_prompt: str, max_ledger_tokens: int, max_answer_tokens: int, args):
    device = model_device(model)
    prompt_text = maybe_apply_chat_template(tokenizer, ledger_prompt, args)
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    prefill_start = time.time()
    generated = model.generate(
        **enc,
        max_new_tokens=int(max_ledger_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ledger_sec = float(time.time() - prefill_start)
    ledger_ids = generated[0, enc.input_ids.shape[1]:].detach().cpu().tolist()
    ledger_text = tokenizer.decode(ledger_ids, skip_special_tokens=True)

    answer_prompt = ledger_prompt + ledger_text + question_prompt
    answer_prompt = maybe_apply_chat_template(tokenizer, answer_prompt, args)
    enc2 = tokenizer(answer_prompt, return_tensors="pt").to(device)
    answer_start = time.time()
    generated2 = model.generate(
        **enc2,
        max_new_tokens=int(max_answer_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    answer_sec = float(time.time() - answer_start)
    answer_ids = generated2[0, enc2.input_ids.shape[1]:].detach().cpu().tolist()
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    output_text = ledger_prompt + ledger_text + question_prompt + answer_text
    return {
        "prefill_sec": 0.0,
        "decode_sec": float(ledger_sec + answer_sec),
        "prompt_tokens": int(enc.input_ids.shape[1]),
        "ledger_generated_tokens": int(len(ledger_ids)),
        "question_tokens": int(tokenizer(question_prompt, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]),
        "answer_generated_tokens": int(len(answer_ids)),
        "ledger_output": ledger_text,
        "answer_output": answer_text,
        "output": output_text,
        "trace_answer_start_record": None,
    }


def aggregate(rows):
    if not rows:
        return {}
    return {
        "num_samples": int(len(rows)),
        "query_acc": float(np.mean([float(r["query_acc"]) for r in rows])),
        "strict_acc": float(np.mean([1.0 if r["strict_acc"] else 0.0 for r in rows])),
        "format_acc": float(np.mean([1.0 if r["format_ok"] else 0.0 for r in rows])),
        "avg_prefill_sec": float(np.mean([float(r["prefill_sec"]) for r in rows])),
        "avg_decode_sec": float(np.mean([float(r["decode_sec"]) for r in rows])),
        "avg_ledger_generated_tokens": float(np.mean([float(r["ledger_generated_tokens"]) for r in rows])),
        "avg_answer_generated_tokens": float(np.mean([float(r["answer_generated_tokens"]) for r in rows])),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    (output_dir / "config_summary.json").write_text(json.dumps(summarize_config(config), indent=2, sort_keys=True))
    if args.config_only:
        print("[generated_memory_hf] wrote config summary only")
        return

    tokenizer = load_tokenizer(args)
    (output_dir / "tokenizer_summary.json").write_text(
        json.dumps(summarize_tokenizer(tokenizer), indent=2, sort_keys=True)
    )
    if args.tokenizer_only:
        print("[generated_memory_hf] wrote config and tokenizer summaries only")
        return

    dtype = dtype_from_name(args.dtype)
    model, auto_class = load_hf_model(args, dtype, config)
    inventory = inventory_model(model, config)
    inventory["auto_class"] = auto_class
    attention_patcher = maybe_install_hf_attention_backend(model, inventory, args)
    if attention_patcher is not None:
        inventory["hf_attention_backend"] = attention_patcher.summary()
    (output_dir / "attention_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True))
    print("[generated_memory_hf] attention_inventory=" + json.dumps(inventory["counts_by_kind"], sort_keys=True))
    ready_marker = os.environ.get("HF_READY_MARKER", "").strip()
    if ready_marker:
        (output_dir / ready_marker).write_text(
            json.dumps(
                {
                    "model_loaded": True,
                    "auto_class": auto_class,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "device": str(model_device(model)),
                    "attention_counts_by_kind": inventory.get("counts_by_kind", {}),
                    "hf_attention_mode": str(args.hf_attention_mode),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.inventory_only:
        if attention_patcher is not None:
            (output_dir / "hf_attention_backend_summary.json").write_text(
                json.dumps(attention_patcher.summary(), indent=2, sort_keys=True)
            )
        return

    static_start = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_START", "128"))
    static_end = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_END", "512"))
    min_prompt_tokens = int(args.min_prompt_tokens)
    if min_prompt_tokens <= 0:
        if args.hf_attention_mode == "native":
            min_prompt_tokens = int(static_start + static_end + 64)
        else:
            min_prompt_tokens = int(args.hf_sparse_static_prefix + args.hf_sparse_static_suffix + 64)

    rng = random.Random(args.seed)
    samples = []
    max_prompt_len = 0
    max_question_len = 0
    for sample_idx in range(int(args.num_samples)):
        query_positions = choose_query_positions(args.num_entries, args.num_queries, rng)
        prompt = build_ledger_prompt(sample_idx, args.num_entries, args.prefill_filler_repeats)
        prompt, prompt_len, used_filler_repeats = ensure_min_prompt_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            sample_idx=sample_idx,
            num_entries=int(args.num_entries),
            filler_repeats=int(args.prefill_filler_repeats),
            min_prompt_tokens=int(min_prompt_tokens),
        )
        question_prompt = build_question_prompt(
            query_positions,
            answer_prefix_scaffold=bool(args.answer_prefix_scaffold),
        )
        question_len = len(tokenizer(question_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0])
        max_prompt_len = max(max_prompt_len, int(prompt_len))
        max_question_len = max(max_question_len, int(question_len))
        samples.append(
            {
                "sample_idx": int(sample_idx),
                "query_positions": [int(x) for x in query_positions],
                "ledger_prompt": prompt,
                "ledger_prompt_tokens": int(prompt_len),
                "question_prompt": question_prompt,
                "question_prompt_tokens": int(question_len),
                "filler_repeats": int(used_filler_repeats),
            }
        )

    max_ledger_new_tokens = int(args.max_new_tokens)
    if max_ledger_new_tokens <= 0:
        max_ledger_new_tokens = int(
            len(tokenizer(build_ledger_output_stub(args.num_entries), return_tensors="pt").input_ids[0])
            + max(16, int(args.generation_margin_tokens))
        )
    max_answer_new_tokens = int(
        len(tokenizer(build_answer_output_stub(args.num_queries), return_tensors="pt").input_ids[0])
        + max(8, int(args.generation_margin_tokens) // 2)
    )
    print(
        f"[generated_memory_hf] max_prompt_len={max_prompt_len} "
        f"max_ledger_new_tokens={max_ledger_new_tokens} "
        f"max_question_tokens={max_question_len} "
        f"max_answer_new_tokens={max_answer_new_tokens}"
    )

    trace = None
    if args.trace_attention:
        trace = AttentionTrace(model=model, config=config, max_records=int(args.trace_max_records))
        trace.install()

    rows = []
    for sample in samples:
        if args.generation_mode == "manual_cache":
            run = hf_manual_cache_run(
                model=model,
                tokenizer=tokenizer,
                ledger_prompt=sample["ledger_prompt"],
                question_prompt=sample["question_prompt"],
                max_ledger_tokens=max_ledger_new_tokens,
                max_answer_tokens=max_answer_new_tokens,
                args=args,
                trace=trace,
            )
        else:
            run = hf_generate_reprefill_run(
                model=model,
                tokenizer=tokenizer,
                ledger_prompt=sample["ledger_prompt"],
                question_prompt=sample["question_prompt"],
                max_ledger_tokens=max_ledger_new_tokens,
                max_answer_tokens=max_answer_new_tokens,
                args=args,
            )
        eval_result = evaluate_output(run["output"], sample["query_positions"], args.num_entries)
        row = {
            **sample,
            **run,
            **eval_result,
            "model_name": args.model_name,
            "generation_mode": args.generation_mode,
            "use_chat_template": bool(args.use_chat_template),
            "disable_thinking": bool(args.disable_thinking),
            "answer_prefix_scaffold": bool(args.answer_prefix_scaffold),
            "answer_constrained_codebook": bool(args.answer_constrained_codebook),
            "force_max_decode_steps": bool(args.force_max_decode_steps),
        }
        rows.append(row)

    if trace is not None:
        trace.remove()
        (output_dir / "attention_trace.json").write_text(json.dumps(trace.records, indent=2))
    hf_attention_backend_summary = attention_patcher.summary() if attention_patcher is not None else {"mode": "native"}
    (output_dir / "hf_attention_backend_summary.json").write_text(
        json.dumps(hf_attention_backend_summary, indent=2, sort_keys=True)
    )

    summary = {
        **aggregate(rows),
        "model_name": args.model_name,
        "auto_class": auto_class,
        "dtype": args.dtype,
        "hf_attention_mode": args.hf_attention_mode,
        "hf_attention_backend": hf_attention_backend_summary,
        "hf_language_model_only": bool(args.hf_language_model_only),
        "generation_mode": args.generation_mode,
        "use_chat_template": bool(args.use_chat_template),
        "disable_thinking": bool(args.disable_thinking),
        "answer_prefix_scaffold": bool(args.answer_prefix_scaffold),
        "answer_constrained_codebook": bool(args.answer_constrained_codebook),
        "force_max_decode_steps": bool(args.force_max_decode_steps),
        "num_entries": int(args.num_entries),
        "num_queries": int(args.num_queries),
        "attention_counts_by_kind": inventory["counts_by_kind"],
    }
    with (output_dir / "generated_memory_results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("[generated_memory_hf] summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
