#!/usr/bin/env python3
import argparse
import csv
import math
import re
import subprocess
from datetime import date
from pathlib import Path


JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def parse_list_int(text: str):
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"Empty integer list: {text!r}")
    return out


def parse_list_float(text: str):
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError(f"Empty float list: {text!r}")
    return out


def compute_scaled_m(
    n_tokens: int,
    base_m: int,
    ref_tokens: int,
    exponent: float,
    min_m: int,
    max_m: int,
):
    ratio = float(max(1, n_tokens)) / float(max(1, ref_tokens))
    raw = float(base_m) * (ratio ** float(exponent))
    m = int(round(raw))
    m = max(int(min_m), min(int(max_m), m))
    if (m % 2) != 0:
        if m < int(max_m):
            m += 1
        else:
            m -= 1
    return max(2, m)


def compute_visits(n_tokens: int, regime: str, param: float):
    if regime == "linear":
        max_visits = int(math.ceil(float(param) * float(n_tokens)))
    elif regime == "sqrt":
        max_visits = int(math.ceil(float(param) * math.sqrt(float(n_tokens))))
    elif regime == "log":
        max_visits = int(math.ceil(float(param) * math.log2(float(max(2, n_tokens)))))
    else:
        raise ValueError(f"Unsupported regime: {regime}")
    return max(16, min(int(n_tokens), max_visits))


def compute_expand_width(max_visits: int, mode: str, fixed: int):
    if mode == "fixed":
        return max(8, int(fixed))
    # Auto heuristic: increase slowly with budget while capping fanout.
    return max(16, min(128, int(round(math.sqrt(float(max_visits))))))


def make_name(prefix: str, regime: str, n_tokens: int, param: float, split_seed: int):
    if regime == "linear":
        tag = f"{int(round(param * 10000.0)):04d}"  # e.g. 0.03 -> 0300
    else:
        tag = str(param).replace(".", "p")
    return f"{prefix}_{regime}_n{n_tokens}_p{tag}_s{split_seed}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit RetrievalAttention decode-complexity sweeps as Slurm jobs."
    )
    parser.add_argument("--test_script", default="test.sh")
    parser.add_argument("--prefix", default="dcs1")
    parser.add_argument("--out_tsv", default="")

    parser.add_argument("--sizes", default="8192,16384,32768,65536")
    parser.add_argument("--families", default="linear,sqrt,log")
    parser.add_argument("--linear_rates", default="0.01,0.02,0.03,0.05")
    parser.add_argument("--sqrt_coeffs", default="1.0,2.0,3.0,4.0")
    parser.add_argument("--log_coeffs", default="8,12,16,24")
    parser.add_argument("--split_seeds", default="1234")

    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--split_mode", default="stratified", choices=["stratified", "random", "contiguous"])
    parser.add_argument("--parity_sample", type=int, default=256)
    parser.add_argument("--trav_sample", type=int, default=128)

    parser.add_argument("--base_roar_m", type=int, default=32)
    parser.add_argument("--base_roar_l", type=int, default=20)
    parser.add_argument("--base_roar_enhance_l", type=int, default=16)
    parser.add_argument("--m_scale_ref_tokens", type=int, default=8192)
    parser.add_argument("--m_scale_exponent", type=float, default=0.5)
    parser.add_argument("--m_min", type=int, default=16)
    parser.add_argument("--m_max", type=int, default=64)

    parser.add_argument("--cand_mult", type=int, default=2)
    parser.add_argument("--min_visits_ratio", type=float, default=0.5)
    parser.add_argument("--expand_mode", default="auto", choices=["auto", "fixed"])
    parser.add_argument("--expand_width", type=int, default=48)

    parser.add_argument("--token_budget_override", type=int, default=100)
    parser.add_argument("--seed_hub_k", type=int, default=256)
    parser.add_argument("--seed_tail_k", type=int, default=128)

    parser.add_argument("--partition", default="")
    parser.add_argument("--time", default="")
    parser.add_argument("--cpus_per_task", type=int, default=0)
    parser.add_argument("--mem", default="")
    parser.add_argument("--gpus_per_node", type=int, default=0)

    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_jobs(args):
    sizes = sorted(set(parse_list_int(args.sizes)))
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    if not families:
        raise ValueError("No families parsed from --families.")

    family_params = {
        "linear": parse_list_float(args.linear_rates),
        "sqrt": parse_list_float(args.sqrt_coeffs),
        "log": parse_list_float(args.log_coeffs),
    }
    split_seeds = parse_list_int(args.split_seeds)

    jobs = []
    for n_tokens in sizes:
        roar_m = compute_scaled_m(
            n_tokens=n_tokens,
            base_m=args.base_roar_m,
            ref_tokens=args.m_scale_ref_tokens,
            exponent=args.m_scale_exponent,
            min_m=args.m_min,
            max_m=args.m_max,
        )
        for split_seed in split_seeds:
            for regime in families:
                params = family_params.get(regime, [])
                for param in params:
                    max_visits = compute_visits(n_tokens=n_tokens, regime=regime, param=param)
                    min_visits = int(round(args.min_visits_ratio * float(max_visits)))
                    min_visits = max(16, min(max_visits, min_visits))
                    expand_width = compute_expand_width(
                        max_visits=max_visits,
                        mode=args.expand_mode,
                        fixed=args.expand_width,
                    )
                    name = make_name(
                        prefix=args.prefix,
                        regime=regime,
                        n_tokens=n_tokens,
                        param=param,
                        split_seed=split_seed,
                    )
                    jobs.append(
                        {
                            "name": name,
                            "n_tokens": int(n_tokens),
                            "regime": regime,
                            "regime_param": float(param),
                            "max_visits": int(max_visits),
                            "min_visits": int(min_visits),
                            "expand_width": int(expand_width),
                            "cand_mult": int(args.cand_mult),
                            "train_frac": float(args.train_frac),
                            "split_mode": args.split_mode,
                            "split_seed": int(split_seed),
                            "parity_sample": int(args.parity_sample),
                            "trav_sample": int(args.trav_sample),
                            "roar_m": int(roar_m),
                            "roar_l": int(args.base_roar_l),
                            "roar_enhance_l": int(args.base_roar_enhance_l),
                            "m_scale_ref_tokens": int(args.m_scale_ref_tokens),
                            "m_scale_exponent": float(args.m_scale_exponent),
                            "token_budget_override": int(args.token_budget_override),
                            "seed_hub_k": int(args.seed_hub_k),
                            "seed_tail_k": int(args.seed_tail_k),
                        }
                    )
    return jobs


def submit_one(args, test_script: Path, job: dict):
    env = {
        "RECALL_ONLY": "1",
        "RECALL_INPUT_TOKENS": str(job["n_tokens"]),
        "RETRIEVALATTN_VALIDATE_PARITY": "1",
        "RETRIEVALATTN_PARITY_LAYERS": "1",
        "RETRIEVALATTN_PARITY_HEADS": "1",
        "RETRIEVALATTN_PARITY_SAMPLE": str(job["parity_sample"]),
        "RETRIEVALATTN_GRAPH_TRAIN_FRAC": str(job["train_frac"]),
        "RETRIEVALATTN_GRAPH_SPLIT": str(job["split_mode"]),
        "RETRIEVALATTN_GRAPH_SPLIT_SEED": str(job["split_seed"]),
        "RETRIEVALATTN_PARITY_HOLDOUT_ONLY": "1",
        "RETRIEVALATTN_TRAVERSAL_EVAL": "1",
        "RETRIEVALATTN_TRAVERSAL_EVAL_SAMPLE": str(job["trav_sample"]),
        "RETRIEVALATTN_SCORE_MODE": "ip",
        "RETRIEVALATTN_QUERY_MODE": "per_head",
        "RETRIEVALATTN_ROAR_M": str(job["roar_m"]),
        "RETRIEVALATTN_ROAR_L": str(job["roar_l"]),
        "RETRIEVALATTN_ROAR_ENHANCE_L": str(job["roar_enhance_l"]),
        "RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT": "0",
        "RETRIEVALATTN_EXPAND_WIDTH": str(job["expand_width"]),
        "RETRIEVALATTN_MIN_VISITS": str(job["min_visits"]),
        "RETRIEVALATTN_MAX_VISITS": str(job["max_visits"]),
        "RETRIEVALATTN_CAND_MULT": str(job["cand_mult"]),
        "RETRIEVALATTN_SEED_HUB_K": str(job["seed_hub_k"]),
        "RETRIEVALATTN_SEED_TAIL_K": str(job["seed_tail_k"]),
        "TOKEN_BUDGET_OVERRIDE": str(job["token_budget_override"]),
    }
    export_text = "ALL," + ",".join(f"{k}={v}" for k, v in env.items())
    cmd = [
        "sbatch",
        "--job-name",
        str(job["name"]),
        "--output",
        f"slurm-{job['name']}-%j.out",
    ]
    if args.partition:
        cmd.extend(["--partition", args.partition])
    if args.time:
        cmd.extend(["--time", args.time])
    if int(args.cpus_per_task) > 0:
        cmd.extend(["--cpus-per-task", str(args.cpus_per_task)])
    if args.mem:
        cmd.extend(["--mem", args.mem])
    if int(args.gpus_per_node) > 0:
        cmd.extend(["--gpus-per-node", str(args.gpus_per_node)])
    cmd.extend(["--export", export_text, str(test_script)])

    if args.dry_run:
        return {
            "submitted": False,
            "job_id": "dry_run",
            "status": "dry_run",
            "stdout": "",
            "stderr": "",
            "cmd": cmd,
        }

    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out_text = (proc.stdout or "").strip()
    err_text = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {
            "submitted": False,
            "job_id": "",
            "status": "submit_failed",
            "stdout": out_text,
            "stderr": err_text,
            "cmd": cmd,
        }
    match = JOB_ID_RE.search(out_text)
    if not match:
        return {
            "submitted": False,
            "job_id": "",
            "status": "submit_parse_failed",
            "stdout": out_text,
            "stderr": err_text,
            "cmd": cmd,
        }
    return {
        "submitted": True,
        "job_id": match.group(1),
        "status": "submitted",
        "stdout": out_text,
        "stderr": err_text,
        "cmd": cmd,
    }


def write_tsv(rows, out_tsv: Path):
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "name",
        "status",
        "n_tokens",
        "regime",
        "regime_param",
        "max_visits",
        "min_visits",
        "expand_width",
        "cand_mult",
        "train_frac",
        "split_mode",
        "split_seed",
        "parity_sample",
        "trav_sample",
        "roar_m",
        "roar_l",
        "roar_enhance_l",
        "m_scale_ref_tokens",
        "m_scale_exponent",
        "token_budget_override",
        "seed_hub_k",
        "seed_tail_k",
        "log_file",
        "submit_stdout",
        "submit_stderr",
    ]
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    test_script = Path(args.test_script).resolve()
    if not test_script.exists():
        raise FileNotFoundError(f"Missing --test_script: {test_script}")

    jobs = build_jobs(args)
    if args.out_tsv:
        out_tsv = Path(args.out_tsv)
    else:
        today = date.today().isoformat()
        out_tsv = Path("notes") / f"{args.prefix}_jobs_{today}.tsv"

    rows = []
    for idx, job in enumerate(jobs, start=1):
        result = submit_one(args=args, test_script=test_script, job=job)
        job_id = result["job_id"]
        if result["status"] == "submitted":
            log_file = f"slurm-{job['name']}-{job_id}.out"
        elif result["status"] == "dry_run":
            log_file = f"slurm-{job['name']}-<jobid>.out"
        else:
            log_file = ""
        row = dict(job)
        row.update(
            {
                "job_id": job_id,
                "status": result["status"],
                "log_file": log_file,
                "submit_stdout": result["stdout"],
                "submit_stderr": result["stderr"],
            }
        )
        rows.append(row)
        print(
            f"[{idx:03d}/{len(jobs):03d}] {job['name']} "
            f"status={result['status']} job_id={job_id}"
        )
        if args.dry_run:
            print("  cmd:", " ".join(result["cmd"]))

    write_tsv(rows=rows, out_tsv=out_tsv)
    submitted = sum(1 for r in rows if r.get("status") == "submitted")
    print(f"[submit_decode_complexity_sweep] submitted={submitted}/{len(rows)}")
    print(f"[submit_decode_complexity_sweep] jobs_tsv={out_tsv}")


if __name__ == "__main__":
    main()
