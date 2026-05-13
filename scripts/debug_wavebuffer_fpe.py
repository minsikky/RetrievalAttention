import importlib.util

import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU


spec = importlib.util.spec_from_file_location("kmeans", "cache_hub/kmeans.py")
km = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km)

B = 1
H = 8
D = 128
N = 10251
STATIC_TOTAL = 68
N_CENTROIDS = 640
NPROBE = 12
PAGE = 8
BUFFER_SIZE = 96
CACHE_SIZE = 64
CORE = 4

print("threadpool", flush=True)
tp = ThreadPool(CORE)
print("wavebuffer", flush=True)
wb = WaveBufferCPU(B, H, D, NPROBE, 0, PAGE, N_CENTROIDS, BUFFER_SIZE, CACHE_SIZE, CORE, tp.get())

cluster_ids = torch.empty((B * H, NPROBE), dtype=torch.int64, pin_memory=True).contiguous()
ints = [torch.zeros((B * H, BUFFER_SIZE), dtype=torch.int32, pin_memory=True).contiguous() for _ in range(9)]
nums = [torch.zeros((B * H,), dtype=torch.int32, pin_memory=True).contiguous() for _ in range(3)]
wb.set_indices(
    ints[0],
    ints[1],
    ints[2],
    nums[0],
    ints[3],
    ints[4],
    ints[5],
    nums[1],
    ints[6],
    ints[7],
    ints[8],
    nums[2],
    cluster_ids,
)

list_keys = torch.empty((B, H, N - STATIC_TOTAL, D), dtype=torch.bfloat16, pin_memory=True).contiguous()
list_vals = torch.empty_like(list_keys)
off_keys = torch.randn((B * H, N - STATIC_TOTAL, D), dtype=torch.bfloat16, pin_memory=True).contiguous()
off_vals = torch.randn((B * H, N - STATIC_TOTAL, D), dtype=torch.bfloat16, pin_memory=True).contiguous()
wb.set_kv(list_keys, list_vals, off_keys, off_vals)

print("kmeans", flush=True)
key = off_keys.cuda()
val = off_vals.cuda()
mean = key.mean(dim=1, keepdim=True)
_, _, clusters, cluster_size = km.segment_k_means(key - mean, val, N_CENTROIDS, 10, 1)
clusters_cpu = clusters.cpu().contiguous()
cluster_size_cpu = cluster_size.cpu().contiguous()
print(
    "cluster stats",
    clusters_cpu.shape,
    cluster_size_cpu.shape,
    int(cluster_size_cpu.min()),
    int(cluster_size_cpu.max()),
    int(cluster_size_cpu.sum()),
    flush=True,
)

print("async", flush=True)
wb.async_construction(clusters_cpu, cluster_size_cpu, 0)
print("sync", flush=True)
wb.construction_sync()
print("done", float(list_keys.float().abs().sum()), flush=True)
