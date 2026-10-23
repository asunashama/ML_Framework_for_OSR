import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np


KEY_METRICS = [
    ("closed_acc", "Closed Acc (%)"),
    ("auroc", "AUROC (%)"),
    ("oscr", "OSCR (%)"),
    ("open_macro_f1", "Open Macro-F1 (%)"),
    ("open_ccr", "Open CCR (%)"),
    ("unknown_rejection_rate", "Unknown Rej (%)"),
    ("known_detection_rate", "Known Det (%)"),
]


def _collect_metrics(exp_dir: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    pattern = os.path.join(exp_dir, "**", "metrics.json")
    files = sorted(glob.glob(pattern, recursive=True))
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rel_parts = os.path.relpath(os.path.dirname(path), exp_dir).replace("\\", "/").split("/")
        method_tag = rel_parts[0] if len(rel_parts) >= 1 else data.get("metadata", {}).get("method", "unknown")
        n_tag = (
            rel_parts[1]
            if len(rel_parts) >= 3
            else f"n{data.get('metadata', {}).get('train_samples_per_class', 'all')}"
        )
        grouped.setdefault((method_tag, n_tag), []).append(data)
    return grouped


def _fmt_mean_std(vals: List[float], scale: float = 100.0, latex: bool = False) -> str:
    arr = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=np.float64) * scale
    if len(arr) == 0:
        return "N/A"
    if len(arr) == 1:
        return f"{arr[0]:.2f}"
    mean, std = float(np.mean(arr)), float(np.std(arr))
    return f"{mean:.2f} $\\pm$ {std:.2f}" if latex else f"{mean:.2f} ± {std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate OSR runs into Markdown, CSV, and LaTeX tables.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory under runs/, e.g. runs/comp_main_benchmark")
    args = parser.parse_args()

    grouped = _collect_metrics(args.exp_dir)
    if not grouped:
        print(f"No metrics.json files found under {args.exp_dir}")
        return

    # 1. Markdown Table
    headers = ["Method", "Train/Cls", "Runs"] + [label for _, label in KEY_METRICS]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    tex_lines = [
        "\\begin{tabular}{llc" + "c" * len(KEY_METRICS) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    csv_lines = [",".join(headers)]

    for (method_tag, n_tag), records in sorted(grouped.items()):
        n_runs = str(len(records))
        md_row = [method_tag, n_tag, n_runs]
        tex_row = [method_tag.replace("_", "\\_"), n_tag.replace("_", "\\_"), n_runs]
        csv_row = [method_tag, n_tag, n_runs]
        for key, _ in KEY_METRICS:
            vals = [r.get(key, float("nan")) for r in records]
            md_row.append(_fmt_mean_std(vals, latex=False))
            tex_row.append(_fmt_mean_std(vals, latex=True))
            csv_row.append(_fmt_mean_std(vals, latex=False))
        md_lines.append("| " + " | ".join(md_row) + " |")
        tex_lines.append(" & ".join(tex_row) + " \\\\")
        csv_lines.append(",".join(csv_row))

    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    md_text = "\n".join(md_lines) + "\n"
    tex_text = "\n".join(tex_lines) + "\n"
    csv_text = "\n".join(csv_lines) + "\n"

    print("\n" + md_text)

    with open(os.path.join(args.exp_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(os.path.join(args.exp_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex_text)
    with open(os.path.join(args.exp_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(csv_text)
    print(f"Saved summary.md, summary.tex, and summary.csv to {args.exp_dir}")


if __name__ == "__main__":
    main()
