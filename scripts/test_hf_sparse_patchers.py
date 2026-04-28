from types import SimpleNamespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmark.generated_memory_hf_eval import (
    inventory_model,
    maybe_install_hf_attention_backend,
)


def make_args(mode: str):
    return SimpleNamespace(
        hf_attention_mode=mode,
        hf_sparse_topk=4,
        hf_sparse_static_prefix=2,
        hf_sparse_static_suffix=2,
        hf_graph_degree=3,
        hf_graph_visit_budget=8,
        hf_graph_seed_count=4,
        hf_graph_online_edges=3,
        hf_graph_search_backend="cpp",
        hf_graph_candidate_target=6,
        hf_graph_expand_width=4,
        hf_graph_min_visits=2,
        hf_graph_frontier_topn=8,
        hf_graph_stop_patience=1,
        hf_graph_stop_margin=0.0,
        hf_graph_roar_cand_limit=8,
        hf_graph_roar_enhance_limit=8,
        hf_graph_roar_entry="hub",
        hf_graph_roar_threads=0,
    )


def make_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model, config


def make_qwen35_model():
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3_5ForCausalLM(config)
    model.eval()
    return model, config


@torch.inference_mode()
def run_mode(mode: str, model_factory=make_model, expected_installed=2):
    torch.manual_seed(1234)
    model, config = model_factory()
    inventory = inventory_model(model, config)
    assert inventory["replacement_plan"]["replaceable_full_attention_count"] == int(expected_installed)
    patcher = maybe_install_hf_attention_backend(model, inventory, make_args(mode))

    input_ids = torch.randint(3, 64, (1, 12), dtype=torch.long)
    out = model(input_ids=input_ids, use_cache=True)
    assert out.logits.shape == (1, 12, config.vocab_size)
    past = out.past_key_values

    for _ in range(3):
        next_id = torch.argmax(out.logits[:, -1:, :], dim=-1)
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        assert out.logits.shape == (1, 1, config.vocab_size)

    if mode != "native":
        summary = patcher.summary()
        assert summary["installed_count"] == int(expected_installed)
        assert summary["decode_calls"] == int(3 * expected_installed)
        assert summary["keep_ratio"] is not None
        if mode in {"graph_topk", "graph_topk_roar"}:
            assert summary["graph_builds"] == int(expected_installed)
            assert summary["visited_total"] > 0
    return True


def main():
    for mode in ("native", "oracle_topk", "graph_topk", "graph_topk_roar"):
        run_mode(mode)
        print(f"{mode}: ok")
    run_mode("oracle_topk", model_factory=make_qwen35_model, expected_installed=2)
    print("qwen3_5 oracle_topk: ok")


if __name__ == "__main__":
    main()
