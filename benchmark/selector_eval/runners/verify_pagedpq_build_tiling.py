#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (  # noqa: E402
    GPUIndex,
    PagePQ,
    build_page_pq_torch,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (  # noqa: E402
    value_vpq_code_stat_risk_torch,
    value_vpq_pack_torch,
    vpq_values_for_tokens_gpu,
)


@dataclass(frozen=True)
class KeyCase:
    name: str
    pages: int
    page_size: int
    dim: int
    subvecs: int
    subbits: int
    dynamic_start: int = 0
    kmeans_iters: int = 3


@contextmanager
def build_tile_env(tile_batch: int):
    names = ("PAGEDPQ_BUILD_TILE_BATCH", "PAGEDPQ_BUILD_TEMP_BUDGET_MB")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["PAGEDPQ_BUILD_TILE_BATCH"] = str(int(tile_batch))
    os.environ["PAGEDPQ_BUILD_TEMP_BUDGET_MB"] = "512"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def legacy_cuda_scatter_env(enabled: bool):
    name = "PAGEDPQ_BUILD_DIAGNOSTIC_LEGACY_ATOMICS"
    previous = os.environ.get(name)
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def seeded_tensor(shape: tuple[int, ...], *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, dtype=torch.float16, device=device, generator=generator)


def clone_tensor_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in outputs.items()}


def tensor_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        return False
    actual_bytes = actual.detach().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(actual_bytes, expected_bytes))


def compare_outputs(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> tuple[bool, str]:
    if actual.keys() != expected.keys():
        return False, f"fields differ: actual={sorted(actual)} expected={sorted(expected)}"
    for name in expected:
        if not tensor_bits_equal(actual[name], expected[name]):
            return False, f"byte mismatch in {name}"
    return True, "all tensor bytes equal"


def _max_float_ulp_difference(
    actual: np.ndarray,
    expected: np.ndarray,
    differing: np.ndarray,
) -> int | None:
    uint_dtype_by_float = {
        np.dtype(np.float16): np.dtype(np.uint16),
        np.dtype(np.float32): np.dtype(np.uint32),
        np.dtype(np.float64): np.dtype(np.uint64),
    }
    uint_dtype = uint_dtype_by_float.get(actual.dtype)
    if uint_dtype is None:
        return None
    bits = int(actual.dtype.itemsize * 8)
    sign_mask = np.array(1 << (bits - 1), dtype=uint_dtype)
    actual_bits = actual.reshape(-1).view(uint_dtype)
    expected_bits = expected.reshape(-1).view(uint_dtype)
    actual_ordered = np.where(
        (actual_bits & sign_mask) != 0,
        ~actual_bits,
        actual_bits ^ sign_mask,
    )
    expected_ordered = np.where(
        (expected_bits & sign_mask) != 0,
        ~expected_bits,
        expected_bits ^ sign_mask,
    )
    upper = np.maximum(actual_ordered, expected_ordered)
    lower = np.minimum(actual_ordered, expected_ordered)
    return int(np.max((upper - lower)[differing]))


def field_difference_summary(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    if actual.dtype != expected.dtype:
        return False, f"dtype={actual.dtype} expected_dtype={expected.dtype}"
    if tuple(actual.shape) != tuple(expected.shape):
        return False, f"shape={tuple(actual.shape)} expected_shape={tuple(expected.shape)}"
    actual_np = np.ascontiguousarray(actual.detach().cpu().numpy())
    expected_np = np.ascontiguousarray(expected.detach().cpu().numpy())
    actual_element_bytes = actual_np.view(np.uint8).reshape(-1, actual_np.dtype.itemsize)
    expected_element_bytes = expected_np.view(np.uint8).reshape(-1, expected_np.dtype.itemsize)
    differing = np.any(actual_element_bytes != expected_element_bytes, axis=1)
    differing_count = int(np.count_nonzero(differing))
    total = int(actual_np.size)
    if differing_count == 0:
        return True, f"differing=0/{total} max_abs=0 max_ulp=0"

    max_abs_text = "n/a"
    if np.issubdtype(actual_np.dtype, np.number):
        with np.errstate(invalid="ignore", over="ignore"):
            abs_difference = np.abs(
                actual_np.reshape(-1).astype(np.float64)
                - expected_np.reshape(-1).astype(np.float64)
            )[differing]
        if bool(np.any(~np.isnan(abs_difference))):
            max_abs_text = f"{float(np.nanmax(abs_difference)):.9g}"
        else:
            max_abs_text = "nan"
    max_ulp = _max_float_ulp_difference(actual_np, expected_np, differing)
    max_ulp_text = str(max_ulp) if max_ulp is not None else "n/a"
    return (
        False,
        f"differing={differing_count}/{total} "
        f"max_abs={max_abs_text} max_ulp={max_ulp_text}",
    )


def compare_outputs_detailed(
    *,
    comparison: str,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    fields: tuple[str, ...],
) -> bool:
    print(f"[determinism] comparison={comparison}")
    passed = True
    for field in fields:
        if field not in actual or field not in expected:
            equal = False
            detail = "field missing"
        else:
            equal, detail = field_difference_summary(actual[field], expected[field])
        print(f"  field={field} result={'PASS' if equal else 'FAIL'} {detail}")
        passed &= equal
    return passed


def build_key_outputs(
    case: KeyCase,
    keys: torch.Tensor,
    *,
    device: torch.device,
    tile_batch: int,
) -> dict[str, torch.Tensor]:
    with build_tile_env(tile_batch):
        index = build_page_pq_torch(
            keys,
            dynamic_start=int(case.dynamic_start),
            indexed_end=int(case.dynamic_start + case.pages * case.page_size),
            page_size=int(case.page_size),
            subvecs=int(case.subvecs),
            subbits=int(case.subbits),
            kmeans_iters=int(case.kmeans_iters),
            seed=1777,
            key_bytes=2,
            router_enabled=False,
            router_prototypes=0,
            router_merge_rel=0.0,
            router_merge_var=0.0,
            router_max_groups=0,
            device=device,
        )
    assert index.native_codebooks is not None
    assert index.native_codes is not None
    assert index.native_page_starts is not None
    return {
        "native_codebooks": index.native_codebooks,
        "native_codes": index.native_codes,
        "native_page_starts": index.native_page_starts,
        "page_codebooks": torch.stack([page.codebooks for page in index.pages]),
        "page_codes": torch.stack([page.codes for page in index.pages]),
        "index_metadata": torch.tensor(
            [index.pending_start, index.indexed_end, index.build_read_mb, index.build_write_mb],
            dtype=torch.float64,
            device=device,
        ),
    }


def dummy_selection_index(
    *,
    pages: int,
    page_size: int,
    dynamic_start: int,
    device: torch.device,
) -> GPUIndex:
    page_list = [
        PagePQ(
            start=int(dynamic_start + page_id * page_size),
            size=int(page_size),
            codebooks=torch.empty((0,), dtype=torch.float32, device=device),
            codes=torch.empty((page_size, 1), dtype=torch.uint8, device=device),
        )
        for page_id in range(pages)
    ]
    return GPUIndex(
        pages=page_list,
        pending_start=int(dynamic_start + pages * page_size),
        indexed_end=int(dynamic_start + pages * page_size),
        build_seconds=0.0,
        build_read_mb=0.0,
        build_write_mb=0.0,
    )


def build_vpq_outputs(
    values: torch.Tensor,
    *,
    pages: int,
    page_size: int,
    dynamic_start: int,
    device: torch.device,
    tile_batch: int,
    kmeans_iters: int = 3,
) -> dict[str, torch.Tensor]:
    index = dummy_selection_index(
        pages=pages,
        page_size=page_size,
        dynamic_start=dynamic_start,
        device=device,
    )
    with build_tile_env(tile_batch):
        pack = value_vpq_pack_torch(
            index=index,
            values=values,
            value_subvecs=1,
            value_subbits=4,
            key_bytes=2,
            device=device,
            kmeans_iters=int(kmeans_iters),
        )
    assert pack is not None
    codebooks, codes, page_starts, packed_page_size, actual_subbits = pack
    build_stats = getattr(index, "_last_value_vpq_build_stats", None)
    assert build_stats is not None
    tokens = torch.arange(int(values.shape[0]), dtype=torch.long, device=device)
    vhat, valid, page_ids, reconstructed_subbits = vpq_values_for_tokens_gpu(
        index=index,
        values=values,
        values_np=None,
        tokens=tokens,
        subbits=8,
        value_subvecs=1,
        value_subbits=4,
        prefer_torch=True,
        value_bytes=2,
        kmeans_iters=int(kmeans_iters),
    )
    residual = values.float() - vhat.float()
    code_error, stat_subbits = value_vpq_code_stat_risk_torch(
        index=index,
        values=values,
        vhat_all=vhat,
        residual_all=residual,
        valid=valid,
        page_ids=page_ids,
        subbits=8,
        value_subvecs=1,
        value_subbits=4,
        value_bytes=2,
        kmeans_iters=int(kmeans_iters),
    )
    scalar_metadata = torch.tensor(
        [packed_page_size, actual_subbits, reconstructed_subbits, stat_subbits],
        dtype=torch.int64,
        device=device,
    )
    return {
        "codebooks": codebooks,
        "codes": codes,
        "page_starts": page_starts,
        "vhat": vhat,
        "residual": residual,
        "code_error": code_error,
        "valid": valid,
        "page_ids": page_ids,
        "scalar_metadata": scalar_metadata,
        "build_io_mb": torch.tensor(build_stats[1:], dtype=torch.float64, device=device),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_comparisons(
    *,
    case_name: str,
    full_batch: int,
    build,
    rows: list[tuple[str, str, str, str, float]],
) -> bool:
    print(f"[verify] {case_name} tile=full", flush=True)
    started = time.perf_counter()
    reference = clone_tensor_outputs(build(full_batch))
    synchronize(next(iter(reference.values())).device)
    rows.append((case_name, "full", "PASS", "reference", time.perf_counter() - started))
    passed = True
    for label, tile_batch in (("1", 1), ("2", 2), ("3", 3), ("7", 7), ("auto", 0)):
        print(f"[verify] {case_name} tile={label}", flush=True)
        started = time.perf_counter()
        try:
            actual = build(tile_batch)
            equal, detail = compare_outputs(actual, reference)
            synchronize(next(iter(actual.values())).device)
        except Exception as exc:  # Keep the table complete before failing the process.
            equal = False
            detail = f"{type(exc).__name__}: {exc}"
        rows.append(
            (
                case_name,
                label,
                "PASS" if equal else "FAIL",
                detail,
                time.perf_counter() - started,
            )
        )
        passed &= equal
    return passed


def print_table(rows: list[tuple[str, str, str, str, float]]) -> None:
    print("case                 tile   result  seconds  detail")
    print("-------------------  -----  ------  -------  ------------------------")
    for case_name, tile, result, detail, seconds in rows:
        print(f"{case_name:19}  {tile:5}  {result:6}  {seconds:7.2f}  {detail}")


def run_determinism_case(
    *,
    case_name: str,
    kmeans_iters: int,
    full_tile: int,
    tiled_tile: int,
    build,
    fields: tuple[str, ...],
    device: torch.device,
) -> bool:
    outputs: dict[int, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
    try:
        for tile_batch in (full_tile, tiled_tile):
            runs = []
            for run_id in (1, 2):
                print(
                    f"[determinism] build={case_name} iters={kmeans_iters} "
                    f"tile={tile_batch} run={run_id}",
                    flush=True,
                )
                started = time.perf_counter()
                run = build(tile_batch)
                synchronize(device)
                print(f"[determinism] build_seconds={time.perf_counter() - started:.2f}")
                runs.append(run)
            outputs[tile_batch] = (runs[0], runs[1])
    except Exception as exc:
        print(
            f"[determinism] build={case_name} iters={kmeans_iters} "
            f"result=FAIL error={type(exc).__name__}: {exc}"
        )
        return False

    same_full = compare_outputs_detailed(
        comparison=f"{case_name}/iters={kmeans_iters}/same_tile={full_tile}",
        actual=outputs[full_tile][1],
        expected=outputs[full_tile][0],
        fields=fields,
    )
    same_tiled = compare_outputs_detailed(
        comparison=f"{case_name}/iters={kmeans_iters}/same_tile={tiled_tile}",
        actual=outputs[tiled_tile][1],
        expected=outputs[tiled_tile][0],
        fields=fields,
    )
    cross_tile = compare_outputs_detailed(
        comparison=f"{case_name}/iters={kmeans_iters}/tile={full_tile}_vs_{tiled_tile}",
        actual=outputs[tiled_tile][0],
        expected=outputs[full_tile][0],
        fields=fields,
    )
    same_config = same_full and same_tiled
    if not same_config:
        verdict = "NONDETERMINISTIC"
    elif not cross_tile:
        verdict = "DETERMINISTIC_TILE_DEPENDENCE"
    else:
        verdict = "BYTE_IDENTICAL"
    print(f"[determinism] verdict={case_name}/iters={kmeans_iters}: {verdict}")
    return same_config and cross_tile


def run_determinism_suite(device: torch.device, *, suite_name: str) -> bool:
    passed = True
    key_case = KeyCase(
        "key_real_23p",
        pages=23,
        page_size=5632,
        dim=128,
        subvecs=4,
        subbits=8,
        dynamic_start=128,
    )
    keys = seeded_tensor(
        (key_case.dynamic_start + key_case.pages * key_case.page_size, key_case.dim),
        seed=12001,
        device=device,
    )
    for kmeans_iters in (3, 1):
        iter_case = replace(key_case, kmeans_iters=kmeans_iters)
        passed &= run_determinism_case(
            case_name=f"{suite_name}/{key_case.name}",
            kmeans_iters=kmeans_iters,
            full_tile=key_case.pages * key_case.subvecs,
            tiled_tile=28,
            build=lambda tile_batch, iter_case=iter_case: build_key_outputs(
                iter_case,
                keys,
                device=device,
                tile_batch=tile_batch,
            ),
            fields=("native_codebooks", "native_codes", "native_page_starts"),
            device=device,
        )
    del keys

    vpq_pages = 23
    vpq_page_size = 5632
    vpq_dynamic_start = 128
    values = seeded_tensor(
        (vpq_dynamic_start + vpq_pages * vpq_page_size, 128),
        seed=13001,
        device=device,
    )
    for kmeans_iters in (3, 1):
        passed &= run_determinism_case(
            case_name=f"{suite_name}/vpq_real_23p",
            kmeans_iters=kmeans_iters,
            full_tile=vpq_pages,
            tiled_tile=7,
            build=lambda tile_batch, kmeans_iters=kmeans_iters: build_vpq_outputs(
                values,
                pages=vpq_pages,
                page_size=vpq_page_size,
                dynamic_start=vpq_dynamic_start,
                device=device,
                tile_batch=tile_batch,
                kmeans_iters=kmeans_iters,
            ),
            fields=("codebooks", "codes", "page_starts", "vhat", "residual", "code_error"),
            device=device,
        )
    print(f"[determinism] suite={suite_name}: {'PASS' if passed else 'FAIL'}")
    return passed


def run_determinism_mode(device: torch.device) -> bool:
    if device.type == "cuda":
        print("[determinism] running historical legacy_scatter diagnostic", flush=True)
        with legacy_cuda_scatter_env(True):
            legacy_passed = run_determinism_suite(device, suite_name="legacy_scatter")
        print(
            f"[determinism] historical_legacy_scatter="
            f"{'BYTE_IDENTICAL' if legacy_passed else 'FAILED'}"
        )
        print("[determinism] running fixed row_mm suite", flush=True)
        with legacy_cuda_scatter_env(False):
            fixed_passed = run_determinism_suite(device, suite_name="row_mm")
    else:
        with legacy_cuda_scatter_env(False):
            fixed_passed = run_determinism_suite(device, suite_name="cpu_historical")
    print(f"DETERMINISM OVERALL: {'PASS' if fixed_passed else 'FAIL'}")
    return fixed_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify byte-identical tiled paged-PQ builds.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("equality", "determinism"), default="equality")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    torch.manual_seed(0)
    if args.mode == "determinism":
        return 0 if run_determinism_mode(device) else 1

    rows: list[tuple[str, str, str, str, float]] = []
    passed = True
    key_cases = (
        KeyCase(
            "key_real_23p",
            pages=23,
            page_size=5632,
            dim=128,
            subvecs=4,
            subbits=8,
            dynamic_start=128,
        ),
        KeyCase(
            "key_one_page_pad",
            pages=1,
            page_size=7,
            dim=20,
            subvecs=4,
            subbits=4,
            dynamic_start=3,
        ),
        KeyCase(
            "key_two_pages_odd",
            pages=2,
            page_size=19,
            dim=20,
            subvecs=4,
            subbits=4,
            dynamic_start=5,
        ),
    )
    for case_id, case in enumerate(key_cases):
        keys = seeded_tensor(
            (case.dynamic_start + case.pages * case.page_size, case.dim),
            seed=12001 + case_id,
            device=device,
        )
        passed &= run_comparisons(
            case_name=case.name,
            full_batch=int(case.pages * case.subvecs),
            build=lambda tile_batch, case=case, keys=keys: build_key_outputs(
                case,
                keys,
                device=device,
                tile_batch=tile_batch,
            ),
            rows=rows,
        )
        del keys

    vpq_pages = 23
    vpq_page_size = 5632
    vpq_dynamic_start = 128
    values = seeded_tensor(
        (vpq_dynamic_start + vpq_pages * vpq_page_size, 128),
        seed=13001,
        device=device,
    )
    passed &= run_comparisons(
        case_name="vpq_real_23p",
        full_batch=vpq_pages,
        build=lambda tile_batch: build_vpq_outputs(
            values,
            pages=vpq_pages,
            page_size=vpq_page_size,
            dynamic_start=vpq_dynamic_start,
            device=device,
            tile_batch=tile_batch,
        ),
        rows=rows,
    )

    print_table(rows)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
