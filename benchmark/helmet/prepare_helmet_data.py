"""Convert one HELMET dataset into the RULER-style jsonl our runner consumes.

HELMET owns prompt construction (templates, shots, middle-truncation to
input_max_length via its tokenize()); we decode the resulting token ids back
to text and let the runner re-tokenize (BOS is stripped here and re-added
there — token count may shift by ±1, harmless). Metrics are computed later
by eval_helmet_preds.py, which reloads the same data deterministically
(same config + seed) and applies HELMET's post_process per sample.

Run under .venv (needs datasets/transformers; no GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELMET_DIR = PROJECT_ROOT / "third_party" / "benchmarks" / "HELMET"


def load_helmet_data(args):
    sys.path.insert(0, str(HELMET_DIR))
    os.chdir(HELMET_DIR)  # HELMET test_files are repo-relative
    from data import load_data  # noqa: E402

    ns = SimpleNamespace(
        max_test_samples=int(args.max_test_samples),
        shots=int(args.shots),
        seed=int(args.seed),
    )
    return load_data(ns, args.dataset, path=args.test_file or None, demo_path=args.demo_file or None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test_file", default="")
    parser.add_argument("--demo_file", default="")
    parser.add_argument("--input_max_length", type=int, required=True)
    parser.add_argument("--generation_max_length", type=int, required=True)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--max_test_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    tok_path = Path(args.tokenizer)
    if tok_path.exists() and not tok_path.is_absolute():
        args.tokenizer = str(tok_path.resolve())

    data = load_helmet_data(args)
    from model_utils import tokenize  # noqa: E402 (HELMET dir on sys.path)
    from transformers import AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i, example in enumerate(data["data"]):
            example = dict(example)
            tokenized = tokenize(
                example,
                data,
                tokenizer=tokenizer,
                max_length=int(args.input_max_length),
                generation_max_length=int(args.generation_max_length),
                use_chat_template=bool(args.use_chat_template),
            )
            text = tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)
            answer = example.get("answer", example.get("answers", ""))
            if isinstance(answer, str):
                outputs = [answer]
            elif isinstance(answer, (list, tuple)):
                outputs = [str(a) for a in answer]
            else:
                outputs = [str(answer)]
            row = {
                "index": i,
                "input": text,
                "outputs": outputs,
                "length": int(tokenized.input_ids.size(1)),
                "others": {"helmet_dataset": args.dataset},
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prepare_helmet_data] wrote {n} rows -> {out_path}")
    # Record the exact dataset args so the eval step can reload the SAME
    # dataset regardless of the evaluating job's environment (a reused
    # data file scored under different test_file joins wrong gold rows).
    meta = {
        "dataset": args.dataset,
        "test_file": args.test_file,
        "demo_file": args.demo_file,
        "shots": int(args.shots),
        "max_test_samples": int(args.max_test_samples),
        "seed": int(args.seed),
        "input_max_length": int(args.input_max_length),
        "generation_max_length": int(args.generation_max_length),
        "rows": n,
    }
    (out_path.parent / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
