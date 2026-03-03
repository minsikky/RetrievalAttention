#!/bin/bash
#SBATCH --job-name=fa_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --account=zhengya98
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1

set -euo pipefail

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

if [ ! -f ".venv/bin/activate" ]; then
  echo "[ERROR] .venv/bin/activate not found."
  exit 1
fi
source .venv/bin/activate

export RETRIEVALATTN_FA_REQUIRE_NATIVE=1

python - <<'PY'
import torch
import flash_attn_2_cuda as ext
from flash_attn import flash_attn_with_kvcache_retrieval

print("[INFO] torch:", torch.__version__)
print("[INFO] torch.cuda:", torch.version.cuda)
print("[INFO] cuda device:", torch.cuda.get_device_name(0))
print("[INFO] has fwd_kvcache_retrieval:", hasattr(ext, "fwd_kvcache_retrieval"))

if not hasattr(ext, "fwd_kvcache_retrieval"):
    raise RuntimeError("flash_attn_2_cuda missing fwd_kvcache_retrieval symbol")

B, S, Hq, Hk, D, K = 1, 64, 32, 8, 128, 32
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.float16)

out, idx, profile = flash_attn_with_kvcache_retrieval(
    q=q,
    k_cache=k,
    v_cache=v,
    causal=True,
    retrieval_topk=K,
    retrieval_group_size=Hq // Hk,
    return_retrieval_idx=True,
)
torch.cuda.synchronize()

print("[INFO] out shape:", tuple(out.shape))
print("[INFO] idx shape:", tuple(idx.shape), "dtype:", idx.dtype)
print("[INFO] profile:", profile)

if profile.get("path") != "native_kernel_fused":
    raise RuntimeError(f"Expected native_kernel_fused path, got: {profile}")
valid_shapes = {(B, S, Hq, K), (B, S, Hk, K)}
if tuple(idx.shape) not in valid_shapes:
    raise RuntimeError(
        f"Unexpected retrieval index shape: {tuple(idx.shape)} "
        f"(expected one of {sorted(valid_shapes)})"
    )

print("[OK] flash_attn_with_kvcache_retrieval smoke test passed.")
PY
