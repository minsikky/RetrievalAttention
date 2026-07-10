#!/usr/bin/env python3
"""Physical-line replay of dependency epoch traces (issue #11, Phase 2).

Consumes the per-(qidx, head) epoch-trace npz files produced by
`run_joint_kv_budget_policy_eval.py --epoch_trace_dir` (trace_format_version
>= 2) and replays the realized per-head walk streams against the RTL physical
contract (issue #11 "Phase 1 ACCEPTED" comment — normative):

  - HBM transaction: 32 B. `requests` counts 32 B transactions (integers).
  - K exact rows: 128 B/token/plane, 128 B-aligned, 4 tx/row.
    TIER_HI reads planes A+B (256 B/token), TIER_LO plane A only.
  - K-scale: 4 B/token on 32 B-aligned lines (8 tokens/line). Dedupe = a
    single last-owner REGISTER keyed {line, slot, tier} in stream order —
    NOT a cache: one intervening different key evicts. (Interpretation
    pinned here: `slot` = the scale-region slot, constant within a lane's
    K-scale stream, so the realized key is (region_slot, line, tier); an
    intervening different line OR tier re-fetches. Flagged for RTL review
    in the summary.)
  - V exact rows: 2 planes x 128 B/token + 4 B/token error sidecars
    (code-error + int8-error) = 260 B/token of committed V. Sidecars are
    4 B/token on 32 B lines (8 tokens/line) and get the same last-owner
    register treatment as K-scale (their own slot).
  - V-PQ code metadata: 5 b/token ({vcode:4, vcommit:1}), ceil(5N/8) B
    sequential coalesced stream.
  - Rounding: each distinct physical line touched rounds up to 32 B tx.
  - Authority split: gather items (K rows, K-scale, V rows/sidecars) use
    the rules above; scan/codebook/metadata streams keep the Phase-1 /
    ladder byte accounting (reconstructed from npz constants below) and
    are charged as sequential 32 B-line streams.
  - Lookahead/rereads stay as fields on the charging epoch (the walk
    segment that issued them) — exactly as the Phase-1 records already
    bucket them. Cross-head dedupe happens ONLY in replay.

Replay matrix per position: line-addressed LRU per KV lane at
{0 B, 64 KiB, 256 KiB, 1 MiB, unlimited} x order {head-serial,
4-head-interleaved}. 0 B = current RTL (no data cache; only the scale/sidecar
last-owner registers). Unlimited = the oracle. OQ=64/path is context for the
RTL max-plus tool, not simulated here.

Order semantics: head-serial = the 4 query heads of a group run back-to-back
(current RTL). 4-head-interleaved = a cross-head gather scheduler: scans run
in lockstep (line-granular round-robin over the shared code/codebook stream),
then epoch i of each head issues token bursts round-robin (one token's line
burst stays contiguous). Interleaving at TOKEN granularity is what lets the
GQA row overlap (M2 union factors) land inside a small reuse window; epoch-
granular interleaving degenerates to serial at ~1.25 epochs/head.

Outputs (under --out_dir):
  - sweep table CSV: position x window x order -> bytes, requests (+ per
    stream-class breakdown), all integers;
  - per position, two epochs-JSON files in the RTL schema (resource_rates +
    epochs[] with name/depends_on/bytes/work/requests only): oracle
    (unlimited window) and bounded (256 KiB), head-serial order;
  - reconciliation JSON vs Phase-1 logical bytes.

Run `--self_test` for the unit tests (K-scale register evict-on-intervening
case, rounding, LRU monotonicity).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

TX = 32  # HBM transaction bytes (contract)

# line-address regions (disjoint per KV lane)
REG_K_ROW_A = 0
REG_K_ROW_B = 1
REG_K_SCALE = 2
REG_V_ROW_A = 3
REG_V_ROW_B = 4
REG_V_SIDECAR = 5
REG_SCAN_KPQ_CODES = 6
REG_SCAN_KPQ_CODEBOOKS = 7
REG_SCAN_VPQ_META = 8
REG_SCAN_VPQ_CODEBOOKS = 9

STREAM_CLASS = {
    REG_K_ROW_A: "k_rows",
    REG_K_ROW_B: "k_rows",
    REG_K_SCALE: "k_scale",
    REG_V_ROW_A: "v_rows",
    REG_V_ROW_B: "v_rows",
    REG_V_SIDECAR: "v_sidecar",
    REG_SCAN_KPQ_CODES: "scan",
    REG_SCAN_KPQ_CODEBOOKS: "scan",
    REG_SCAN_VPQ_META: "scan",
    REG_SCAN_VPQ_CODEBOOKS: "scan",
}

# frozen-config scan constants (match the golden sim ladder accounting)
PAGE_SIZE = 5632
KPQ_SUBVECS = 4           # --subvecs 4, 1 B codes (subbits 8)
KPQ_CODEBOOK_BYTES = 4 * 256 * 32 * 2   # 64 KiB / page (subvecs x 2^8 x dsub x fp16)
VPQ_CODEBOOK_BYTES = 1 * 16 * 128 * 2   # 4 KiB / page (value_subvecs=1, subbits=4)
VPQ_STAT_BYTES = 1 * 16 * 2             # per-page code-stat table (code_stat_bytes=2)

EVENT_KIND = {0: "start_eval", 1: "k_up", 2: "v_up", 3: "commit"}

# RTL example resource rates (chip-aggregate; from the contract schema)
RESOURCE_RATES = {"scan_items": 1000000000, "qk_flops": 6553600000000}

WINDOWS = [("0B", 0), ("64KiB", 64 * 1024), ("256KiB", 256 * 1024), ("1MiB", 1024 * 1024), ("unlimited", -1)]
ORDERS = ["head_serial", "interleaved"]
BOUNDED_PRIMARY = "256KiB"


class LastOwnerRegister:
    """Single last-owner register for small-grain 4 B/token streams (K-scale,
    V sidecars). One 32 B request per NEW owner key {slot, line, tier} in
    stream order; consecutive same-key accesses reuse; ANY intervening
    different key evicts (explicitly not a cache)."""

    def __init__(self) -> None:
        self.key: tuple | None = None

    def access(self, slot: int, line: int, tier: int) -> bool:
        """Returns True when the access needs a fetch (register miss)."""
        key = (int(slot), int(line), int(tier))
        if self.key == key:
            return False
        self.key = key
        return True


class LineLRU:
    """Line-addressed LRU over 32 B line ids. capacity_bytes < 0 => unlimited;
    0 => no cache (every access misses)."""

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_lines = -1 if capacity_bytes < 0 else capacity_bytes // TX
        self._od: OrderedDict = OrderedDict()
        self._set: set = set()

    def access(self, line_id: tuple) -> bool:
        """Returns True on miss (fetch charged)."""
        if self.capacity_lines == 0:
            return True
        if self.capacity_lines < 0:
            if line_id in self._set:
                return False
            self._set.add(line_id)
            return True
        od = self._od
        if line_id in od:
            od.move_to_end(line_id)
            return False
        od[line_id] = None
        if len(od) > self.capacity_lines:
            od.popitem(last=False)
        return True


def _seq_lines(region: int, n_bytes: int) -> list[tuple]:
    return [(region, i) for i in range(math.ceil(int(n_bytes) / TX))]


class HeadTrace:
    """Parsed per-head epoch trace with per-epoch gather token sets."""

    def __init__(self, npz_path: Path) -> None:
        d = np.load(npz_path, allow_pickle=False)
        if int(d.get("trace_format_version", np.int64(1))) < 2:
            raise ValueError(f"{npz_path}: needs trace_format_version >= 2 (rerun tracer)")
        self.qidx = int(d["qidx"])
        self.head = int(d["head"])
        self.kv_head = int(d["kv_head"])
        self.context_len = int(d["context_len"])
        self.n_pages = int(d["n_pages"])
        self.head_dim = int(d["head_dim"])
        self.n_epochs = int(d["n_epochs"])
        self.kinds = np.asarray(d["epoch_event_kind_code"], dtype=np.int64)
        # Phase-1 marginal-band logical bytes (region arrays; start sets are
        # NOT included there by design) -- kept for the reconciliation gate.
        self.phase1_region_logical = float(np.sum(np.asarray(d["epoch_region_logical_bytes"], dtype=np.float64)))
        k_hi = np.asarray(d["k_hi_tokens"], dtype=np.int64)
        k_hi_start = np.asarray(d["k_hi_tokens_start"], dtype=np.int64)
        if not np.array_equal(np.sort(k_hi), np.sort(k_hi_start)):
            raise AssertionError(f"{npz_path}: K hi set not constant across walk (frozen split violated)")
        self.k_hi_set = frozenset(int(t) for t in k_hi.tolist())
        km, ko = np.asarray(d["k_marginal_tokens"]), np.asarray(d["k_marginal_offsets"])
        vm, vo = np.asarray(d["v_marginal_tokens"]), np.asarray(d["v_marginal_offsets"])
        hb, ho = np.asarray(d["hi_boundary_tokens"]), np.asarray(d["hi_boundary_offsets"])
        self.k_tokens_by_epoch: list[np.ndarray] = []
        self.v_tokens_by_epoch: list[np.ndarray] = []
        self.hib_by_epoch: list[np.ndarray] = []
        for i in range(self.n_epochs):
            k_band = km[ko[i] : ko[i + 1]]
            v_band = vm[vo[i] : vo[i + 1]]
            if i == 0:
                # start_eval carries the start-rung committed read sets plus
                # its own lookahead bands (all charged on this epoch, per the
                # lookahead-stays-on-charging-epoch rule).
                k_band = np.unique(np.concatenate([np.asarray(d["start_k_tokens"], dtype=np.int64), k_band]))
                v_band = np.unique(np.concatenate([np.asarray(d["start_v_tokens"], dtype=np.int64), v_band]))
            self.k_tokens_by_epoch.append(np.sort(np.asarray(k_band, dtype=np.int64)))
            self.v_tokens_by_epoch.append(np.sort(np.asarray(v_band, dtype=np.int64)))
            self.hib_by_epoch.append(np.sort(np.asarray(hb[ho[i] : ho[i + 1]], dtype=np.int64)))
        d.close()

    def epoch_name(self, i: int) -> str:
        return f"h{self.head}_e{i}_{EVENT_KIND[int(self.kinds[i])]}"

    def scan_streams(self) -> list[tuple[int, int]]:
        """(region, bytes) sequential scan streams per head (ladder widths +
        the RTL V-PQ metadata rule)."""
        return [
            (REG_SCAN_KPQ_CODES, self.n_pages * PAGE_SIZE * KPQ_SUBVECS),
            (REG_SCAN_KPQ_CODEBOOKS, self.n_pages * KPQ_CODEBOOK_BYTES),
            (REG_SCAN_VPQ_META, math.ceil(5 * self.context_len / 8)),
            (REG_SCAN_VPQ_CODEBOOKS, self.n_pages * (VPQ_CODEBOOK_BYTES + VPQ_STAT_BYTES)),
        ]


def _epoch_items(ht: HeadTrace, i: int):
    """Yield gather ITEMS for epoch i of a head, in stream order. One item =
    one token's access burst (kept contiguous): list of (kind, payload) with
    kind 'line' -> (region, index) and 'reg' -> (slot_region, line, tier).
    Token streams ascend (gather engine order: K tokens with row lines then
    scale; hi-boundary lifts; then V tokens rows + sidecar). Interleaved
    replay round-robins at ITEM granularity so per-token bursts stay whole."""
    hi = ht.k_hi_set
    for t in ht.k_tokens_by_epoch[i].tolist():
        base = t * 4
        tier = 1 if t in hi else 0
        item = [("line", (REG_K_ROW_A, base + j)) for j in range(4)]
        if tier:
            item += [("line", (REG_K_ROW_B, base + j)) for j in range(4)]
        item.append(("reg", (REG_K_SCALE, t // 8, tier)))
        yield item
    for t in ht.hib_by_epoch[i].tolist():
        # hi-boundary lift: plane B rows only (plane A + scale already read
        # as part of the lo-tier band); scale re-read at hi tier.
        base = t * 4
        item = [("line", (REG_K_ROW_B, base + j)) for j in range(4)]
        item.append(("reg", (REG_K_SCALE, t // 8, 1)))
        yield item
    for t in ht.v_tokens_by_epoch[i].tolist():
        base = t * 4
        item = []
        for j in range(4):
            item.append(("line", (REG_V_ROW_A, base + j)))
            item.append(("line", (REG_V_ROW_B, base + j)))
        item.append(("reg", (REG_V_SIDECAR, t // 8, 0)))
        yield item


def replay_lane(
    heads: list[HeadTrace],
    *,
    window_bytes: int,
    order: str,
) -> dict:
    """Replay one KV lane (the <=4 query heads sharing it) at one window/order.
    Returns per-epoch bytes/requests attribution + lane totals per class."""
    lru = LineLRU(window_bytes)
    regs: dict[int, LastOwnerRegister] = {
        REG_K_SCALE: LastOwnerRegister(),
        REG_V_SIDECAR: LastOwnerRegister(),
    }
    per_epoch: dict[str, dict[str, int]] = {}
    class_bytes: dict[str, int] = defaultdict(int)

    def charge(name: str, region: int, n_tx: int) -> None:
        e = per_epoch.setdefault(name, {"bytes": 0, "requests": 0})
        e["bytes"] += n_tx * TX
        e["requests"] += n_tx
        class_bytes[STREAM_CLASS[region]] += n_tx * TX

    def run_item(name: str, item) -> None:
        per_epoch.setdefault(name, {"bytes": 0, "requests": 0})
        for kind, payload in item:
            if kind == "line":
                if lru.access(payload):
                    charge(name, payload[0], 1)
            else:
                slot, line, tier = payload
                if regs[slot].access(slot, line, tier):
                    if lru.access((slot, line)):
                        charge(name, slot, 1)

    def scan_items(ht: HeadTrace):
        # one item per 32 B line of the sequential scan streams
        name = f"h{ht.head}_pq_scan"
        per_epoch.setdefault(name, {"bytes": 0, "requests": 0})
        for region, n_bytes in ht.scan_streams():
            for line in _seq_lines(region, n_bytes):
                yield name, [("line", line)]

    def epoch_items_named(ht: HeadTrace, i: int):
        name = ht.epoch_name(i)
        per_epoch.setdefault(name, {"bytes": 0, "requests": 0})
        for item in _epoch_items(ht, i):
            yield name, item

    def round_robin(streams) -> None:
        # item-granular round-robin across the group's heads: models a
        # cross-head gather scheduler issuing one token burst per head in
        # turn, so cross-head row overlap lands within a small reuse window.
        active = [iter(s) for s in streams]
        while active:
            nxt = []
            for it in active:
                try:
                    name, item = next(it)
                except StopIteration:
                    continue
                run_item(name, item)
                nxt.append(it)
            active = nxt

    if order == "head_serial":
        for ht in heads:
            for name, item in scan_items(ht):
                run_item(name, item)
            for i in range(ht.n_epochs):
                for name, item in epoch_items_named(ht, i):
                    run_item(name, item)
    elif order == "interleaved":
        # phase 1: scans in lockstep (the 4 heads walk the same code/codebook
        # stream); phase 2: epoch i of each head, token-burst round-robin.
        round_robin([scan_items(ht) for ht in heads])
        max_ep = max(ht.n_epochs for ht in heads)
        for i in range(max_ep):
            round_robin([epoch_items_named(ht, i) for ht in heads if i < ht.n_epochs])
    else:
        raise ValueError(order)

    return {"per_epoch": per_epoch, "class_bytes": dict(class_bytes)}


def logical_bytes_lane(heads: list[HeadTrace]) -> int:
    """Phase-1-basis logical bytes for the lane (no dedupe, no rounding):
    gather tokens x logical widths (K lo 128 / hi 256 + scale 4; V 260) +
    per-head scan stream bytes. Used for the reconciliation gate."""
    total = 0
    for ht in heads:
        for region, n_bytes in ht.scan_streams():
            total += int(n_bytes)
        for i in range(ht.n_epochs):
            for t in ht.k_tokens_by_epoch[i].tolist():
                total += 256 + 4 if t in ht.k_hi_set else 128 + 4
            total += 128 * len(ht.hib_by_epoch[i])
            total += 260 * len(ht.v_tokens_by_epoch[i])
    return total


def build_epochs_json(heads: list[HeadTrace], replay: dict) -> dict:
    """RTL schema: resource_rates + per-head DAG epochs (name/depends_on/
    bytes/work/requests only; integers; depends_on within head)."""
    epochs = []
    per_epoch = replay["per_epoch"]
    for ht in sorted(heads, key=lambda h: h.head):
        scan_name = f"h{ht.head}_pq_scan"
        se = per_epoch.get(scan_name, {"bytes": 0, "requests": 0})
        epochs.append(
            {
                "name": scan_name,
                "depends_on": [],
                "bytes": int(se["bytes"]),
                "work": {"scan_items": int(ht.context_len)},
                "requests": int(se["requests"]),
            }
        )
        prev = scan_name
        for i in range(ht.n_epochs):
            name = ht.epoch_name(i)
            e = per_epoch.get(name, {"bytes": 0, "requests": 0})
            n_k = len(ht.k_tokens_by_epoch[i]) + len(ht.hib_by_epoch[i])
            n_v = len(ht.v_tokens_by_epoch[i])
            epochs.append(
                {
                    "name": name,
                    "depends_on": [prev],
                    "bytes": int(e["bytes"]),
                    "work": {"qk_flops": int(2 * ht.head_dim * (n_k + n_v))},
                    "requests": int(e["requests"]),
                }
            )
            prev = name
    return {"resource_rates": dict(RESOURCE_RATES), "epochs": epochs}


def validate_epochs_json(doc: dict) -> None:
    assert set(doc.keys()) == {"resource_rates", "epochs"}
    names = set()
    for e in doc["epochs"]:
        assert set(e.keys()) == {"name", "depends_on", "bytes", "work", "requests"}, e.keys()
        assert isinstance(e["bytes"], int) and e["bytes"] >= 0
        assert isinstance(e["requests"], int) and e["requests"] >= 0
        for k, v in e["work"].items():
            assert k in doc["resource_rates"] and isinstance(v, int) and v >= 0
        for dep in e["depends_on"]:
            assert dep in names, f"forward/unknown dep {dep}"
            assert dep.split("_")[0] == e["name"].split("_")[0], "cross-head dep"
        names.add(e["name"])


def replay_position(trace_dir: Path, qidx: int, out_dir: Path) -> list[dict]:
    files = sorted(trace_dir.glob(f"epoch_q{qidx}_h*.npz"), key=lambda p: int(p.stem.split("_h")[1]))
    if not files:
        raise FileNotFoundError(f"no traces for q{qidx} in {trace_dir}")
    heads = [HeadTrace(p) for p in files]
    ctx = heads[0].context_len
    lanes: dict[int, list[HeadTrace]] = defaultdict(list)
    for ht in heads:
        lanes[ht.kv_head].append(ht)
    for lane in lanes.values():
        lane.sort(key=lambda h: h.head)

    dense_bytes_per_head = ctx * heads[0].head_dim * 2 * 2  # fp16 K+V rows
    logical_total = sum(logical_bytes_lane(lane) for lane in lanes.values())
    phase1_region_logical = int(round(sum(ht.phase1_region_logical for ht in heads)))

    rows = []
    json_replays: dict[str, dict] = {}
    prev_bytes: dict[str, int] = {}
    for order in ORDERS:
        for wname, wbytes in WINDOWS:
            lane_results = {kv: replay_lane(lane, window_bytes=wbytes, order=order) for kv, lane in lanes.items()}
            tot_bytes = sum(sum(e["bytes"] for e in r["per_epoch"].values()) for r in lane_results.values())
            tot_req = sum(sum(e["requests"] for e in r["per_epoch"].values()) for r in lane_results.values())
            cls: dict[str, int] = defaultdict(int)
            for r in lane_results.values():
                for c, b in r["class_bytes"].items():
                    cls[c] += b
            # gate: monotone non-increasing bytes as window grows, per order
            if order in prev_bytes and tot_bytes > prev_bytes[order]:
                raise AssertionError(f"q{qidx} {order}: bytes increased {prev_bytes[order]} -> {tot_bytes} at {wname}")
            prev_bytes[order] = tot_bytes
            rows.append(
                {
                    "qidx": qidx,
                    "context_len": ctx,
                    "order": order,
                    "window": wname,
                    "window_bytes": max(0, wbytes),
                    "physical_bytes": int(tot_bytes),
                    "requests": int(tot_req),
                    "bytes_k_rows": int(cls.get("k_rows", 0)),
                    "bytes_k_scale": int(cls.get("k_scale", 0)),
                    "bytes_v_rows": int(cls.get("v_rows", 0)),
                    "bytes_v_sidecar": int(cls.get("v_sidecar", 0)),
                    "bytes_scan": int(cls.get("scan", 0)),
                    "logical_bytes_contract_widths": int(logical_total),
                    "phase1_region_logical_bytes": int(phase1_region_logical),
                    "dense_bytes_32heads": int(dense_bytes_per_head * len(heads)),
                    "physical_over_dense": float(tot_bytes) / float(dense_bytes_per_head * len(heads)),
                    "physical_over_contract_logical": float(tot_bytes) / float(logical_total),
                }
            )
            if order == "head_serial" and wname in ("unlimited", BOUNDED_PRIMARY):
                merged: dict[str, dict[str, int]] = {}
                for r in lane_results.values():
                    merged.update(r["per_epoch"])
                json_replays[wname] = {"per_epoch": merged}

    # oracle == unlimited by construction: emit and assert label equivalence
    oracle_doc = build_epochs_json(heads, json_replays["unlimited"])
    bounded_doc = build_epochs_json(heads, json_replays[BOUNDED_PRIMARY])
    validate_epochs_json(oracle_doc)
    validate_epochs_json(bounded_doc)
    ub = [r for r in rows if r["order"] == "head_serial" and r["window"] == "unlimited"][0]
    assert sum(e["bytes"] for e in oracle_doc["epochs"]) == ub["physical_bytes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"epochs_q{qidx}_oracle.json").write_text(json.dumps(oracle_doc, indent=2), encoding="utf-8")
    (out_dir / f"epochs_q{qidx}_bounded_{BOUNDED_PRIMARY}.json").write_text(
        json.dumps(bounded_doc, indent=2), encoding="utf-8"
    )
    return rows


# ---------------------------------------------------------------------------
# self tests
# ---------------------------------------------------------------------------

def self_test() -> None:
    # K-scale last-owner register: evict-on-intervening-key
    r = LastOwnerRegister()
    assert r.access(REG_K_SCALE, 1, 0) is True      # first touch fetches
    assert r.access(REG_K_SCALE, 1, 0) is False     # consecutive same key reuses
    assert r.access(REG_K_SCALE, 2, 0) is True      # new line fetches
    assert r.access(REG_K_SCALE, 1, 0) is True      # RE-FETCH: intervening key evicted line 1
    assert r.access(REG_K_SCALE, 1, 1) is True      # same line, different TIER = different key
    assert r.access(REG_K_SCALE, 1, 1) is False
    assert r.access(REG_K_SCALE, 1, 0) is True      # tier flap re-fetches (register, not cache)

    # 8 ascending tokens share one scale line -> 1 fetch
    r2 = LastOwnerRegister()
    fetches = sum(r2.access(REG_K_SCALE, t // 8, 0) for t in range(8))
    assert fetches == 1
    # tokens 0..15 -> 2 lines -> 2 fetches
    r3 = LastOwnerRegister()
    assert sum(r3.access(REG_K_SCALE, t // 8, 0) for t in range(16)) == 2

    # LRU: unlimited dedupes, 0B always misses, monotone across sizes
    stream = [(0, i % 10) for i in range(100)] + [(1, i) for i in range(50)]
    misses = {}
    for cap in (0, TX * 8, TX * 32, -1):
        lru = LineLRU(cap)
        misses[cap] = sum(lru.access(x) for x in stream)
    assert misses[0] == len(stream)
    assert misses[-1] == 60  # 10 + 50 distinct lines
    assert misses[0] >= misses[TX * 8] >= misses[TX * 32] >= misses[-1]

    # LRU eviction order: capacity 2 lines, A B C A -> A misses again
    lru = LineLRU(TX * 2)
    a, b, c = (0, 1), (0, 2), (0, 3)
    assert lru.access(a) and lru.access(b) and lru.access(c)
    assert lru.access(a) is True  # A was evicted by C

    # rounding: 5-byte stream -> 1 tx; 33 bytes -> 2 tx
    assert len(_seq_lines(REG_SCAN_VPQ_META, 5)) == 1
    assert len(_seq_lines(REG_SCAN_VPQ_META, 33)) == 2

    print("self_test: ALL PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--trace_dir", help="dir with epoch_q*_h*.npz (format v2)")
    ap.add_argument("--qidx", help="comma-separated qidx list")
    ap.add_argument("--out_dir")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    all_rows: list[dict] = []
    for q in [int(x) for x in str(args.qidx).split(",")]:
        rows = replay_position(trace_dir, q, out_dir)
        all_rows.extend(rows)
        ub = [r for r in rows if r["order"] == "head_serial" and r["window"] == "unlimited"][0]
        zb = [r for r in rows if r["order"] == "head_serial" and r["window"] == "0B"][0]
        print(
            f"q{q} ctx={ub['context_len']}: 0B={zb['physical_bytes']/1e6:.2f}MB "
            f"oracle={ub['physical_bytes']/1e6:.2f}MB dense_ratio(0B)={zb['physical_over_dense']:.4f} "
            f"physical/contract-logical(0B)={zb['physical_over_contract_logical']:.4f}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with (out_dir / "replay_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out_dir}/replay_sweep.csv ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
