import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from osrb.core.config import load_yaml


def _build_job_list(sweep_cfg: Dict[str, Any], runs_root: str, force: bool = False) -> List[Dict[str, Any]]:
    exp_name = sweep_cfg.get("experiment_name", "sweep")
    ds_cfg_path = sweep_cfg["dataset_config"]
    method_cfg_paths = sweep_cfg["method_configs"]
    split_seeds = sweep_cfg.get("split_seeds", [42])
    seeds = sweep_cfg.get("seeds", None)
    train_samples_list = sweep_cfg.get("train_samples_per_class_list", [None])
    extra_overrides = sweep_cfg.get("overrides", [])

    jobs = []
    for m_path in method_cfg_paths:
        m_tag = os.path.splitext(os.path.basename(m_path))[0]
        for tr_n in train_samples_list:
            n_tag = f"n{tr_n}" if tr_n is not None else "nall"
            for sp_seed in split_seeds:
                run_seeds = seeds if seeds is not None else [sp_seed]
                for sd in run_seeds:
                    out_dir = os.path.join(
                        runs_root, exp_name, m_tag, n_tag, f"split{sp_seed}_seed{sd}"
                    )
                    metrics_file = os.path.join(out_dir, "metrics.json")
                    if os.path.exists(metrics_file) and not force:
                        print(f"[Sweep] Skipping completed run: {out_dir}")
                        continue

                    overrides = [f"seed={sd}", f"data.split_seed={sp_seed}"]
                    if tr_n is not None:
                        overrides.append(f"data.train_samples_per_class={tr_n}")
                        if int(tr_n) > 100 and "comp" in ds_cfg_path.lower():
                            overrides.append("data.pool_and_resplit=true")
                    overrides.extend(extra_overrides)

                    jobs.append(
                        {
                            "dataset_config": ds_cfg_path,
                            "method_config": m_path,
                            "output_dir": out_dir,
                            "overrides": overrides,
                            "tag": f"{m_tag}/{n_tag}/split{sp_seed}_seed{sd}",
                        }
                    )
    return jobs


def _worker(gpu_id: int, job_queue: "queue.Queue[Dict[str, Any]]") -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return

        os.makedirs(job["output_dir"], exist_ok=True)
        log_path = os.path.join(job["output_dir"], "train.log")
        cmd = [
            sys.executable,
            os.path.join(ROOT_DIR, "scripts", "train.py"),
            "--dataset-config",
            job["dataset_config"],
            "--method-config",
            job["method_config"],
            "--output-dir",
            job["output_dir"],
            "--override",
            *job["overrides"],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[GPU {gpu_id}] Starting: {job['tag']} -> {log_path}")
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=ROOT_DIR, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        status = "OK" if proc.returncode == 0 else f"FAILED (code={proc.returncode})"
        print(f"[GPU {gpu_id}] Finished: {job['tag']} [{status}] in {dt:.1f}s")
        job_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU parallel experiment sweep runner.")
    parser.add_argument("--sweep-config", type=str, required=True, help="Path to sweep YAML config.")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root directory for run outputs.")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="Override GPU IDs, e.g. --gpus 0 1")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print queued jobs without executing.")
    args = parser.parse_args()

    sweep_cfg = load_yaml(args.sweep_config)
    gpus = args.gpus if args.gpus is not None else sweep_cfg.get("gpus", [0, 1])
    jobs = _build_job_list(sweep_cfg, runs_root=args.runs_root, force=args.force)

    print(f"[Sweep] Total jobs queued: {len(jobs)} across GPUs {gpus}")
    if args.dry_run or not jobs:
        for i, j in enumerate(jobs, 1):
            print(f"  {i:3d}. {j['tag']} | overrides={j['overrides']}")
        return

    q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    for j in jobs:
        q.put(j)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_worker, gpu_id, q) for gpu_id in gpus]
        for fut in futures:
            fut.result()


if __name__ == "__main__":
    main()
