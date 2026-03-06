#!/bin/bash
#SBATCH --job-name=fa_graph_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

set -euo pipefail

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
source .venv/bin/activate

python - <<'PY'
import torch
import flash_attn_2_cuda as ext
from flash_attn import flash_attn_with_kvcache_retrieval_graph

print("[INFO] torch:", torch.__version__)
print("[INFO] torch.cuda:", torch.version.cuda)
print("[INFO] cuda device:", torch.cuda.get_device_name(0))
print("[INFO] has fwd_kvcache_retrieval:", hasattr(ext, "fwd_kvcache_retrieval"))
print("[INFO] has fwd_kvcache_retrieval_graph:", hasattr(ext, "fwd_kvcache_retrieval_graph"))

if not hasattr(ext, "fwd_kvcache_retrieval_graph"):
    raise RuntimeError("flash_attn_2_cuda missing fwd_kvcache_retrieval_graph symbol")

B, S, Hq, Hk, D = 1, 64, 32, 8, 128
K = 32
GRAPH_NQ = 8
GRAPH_DEG = 16
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)

out, idx, graph, profile = flash_attn_with_kvcache_retrieval_graph(
    q=q,
    k_cache=k,
    v_cache=v,
    causal=True,
    retrieval_topk=K,
    retrieval_group_size=Hq // Hk,
    retrieval_normalize=True,
    graph_nq=GRAPH_NQ,
    graph_degree=GRAPH_DEG,
    graph_dynamic_start=0,
    graph_dynamic_end=S,
)
torch.cuda.synchronize()

print("[INFO] out shape:", tuple(out.shape))
print("[INFO] idx shape:", tuple(idx.shape), "dtype:", idx.dtype)
print("[INFO] graph shape:", tuple(graph.shape), "dtype:", graph.dtype)
print("[INFO] profile:", profile)

if profile.get("path") != "native_kernel_fused_graph":
    raise RuntimeError(f"Expected native_kernel_fused_graph path, got: {profile}")

valid_idx_shapes = {(B, S, Hq, K), (B, S, Hk, K)}
if tuple(idx.shape) not in valid_idx_shapes:
    raise RuntimeError(
        f"Unexpected retrieval index shape: {tuple(idx.shape)} "
        f"(expected one of {sorted(valid_idx_shapes)})"
    )

expected_graph_shape = (B, Hk, S, GRAPH_DEG)
if tuple(graph.shape) != expected_graph_shape:
    raise RuntimeError(
        f"Unexpected graph shape: {tuple(graph.shape)} "
        f"(expected {expected_graph_shape})"
    )
if graph.dtype != torch.int32:
    raise RuntimeError(f"Unexpected graph dtype: {graph.dtype}")

print("[OK] flash_attn_with_kvcache_retrieval_graph smoke test passed.")
PY
