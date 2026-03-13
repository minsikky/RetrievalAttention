import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize generated-memory length sweep results")
    parser.add_argument(
        "--root",
        type=str,
        default="generated_memory_eval_result/length_sweep_s16_fullgpu",
        help="Root directory containing per-run summary.json files",
    )
    return parser.parse_args()


def parse_run_dir_name(name: str):
    if "_e" not in name:
        return name, -1
    mode, entries = name.rsplit("_e", 1)
    try:
        return mode, int(entries)
    except ValueError:
        return mode, -1


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        mode, entries_from_dir = parse_run_dir_name(summary_path.parent.name)
        rows.append(
            {
                "mode": mode,
                "num_entries": int(data.get("num_entries", entries_from_dir)),
                "avg_total_generated_tokens": float(data.get("avg_total_generated_tokens", 0.0)),
                "avg_decode_sec": float(data.get("avg_decode_sec", 0.0)),
                "avg_prefill_sec": float(data.get("avg_prefill_sec", 0.0)),
                "query_acc": float(data.get("query_acc", 0.0)),
                "strict_acc": float(data.get("strict_acc", 0.0)),
                "format_acc": float(data.get("format_acc", 0.0)),
                "summary_path": str(summary_path),
            }
        )

    rows.sort(key=lambda row: (row["mode"], row["num_entries"]))
    header = [
        "mode",
        "num_entries",
        "avg_total_generated_tokens",
        "avg_decode_sec",
        "avg_prefill_sec",
        "query_acc",
        "strict_acc",
        "format_acc",
        "summary_path",
    ]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["mode"]),
                    str(row["num_entries"]),
                    f"{row['avg_total_generated_tokens']:.1f}",
                    f"{row['avg_decode_sec']:.3f}",
                    f"{row['avg_prefill_sec']:.3f}",
                    f"{row['query_acc']:.3f}",
                    f"{row['strict_acc']:.3f}",
                    f"{row['format_acc']:.3f}",
                    row["summary_path"],
                ]
            )
        )


if __name__ == "__main__":
    main()
