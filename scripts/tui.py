#!/usr/bin/env python3
"""
Interactive Terminal GUI (TUI) for ML_Framework_for_OSR.
Supports full-screen arrow-key navigation (curses) and clean ANSI interactive prompts.
"""

import argparse
import curses
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from osrb.core.config import load_yaml


# ==============================================================================
# Helper Functions: System, GPU, Configs, and Processes
# ==============================================================================
def get_gpu_status_lines() -> List[str]:
    """Query nvidia-smi for concise GPU status lines."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="replace")
        lines = []
        for row in out.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 6:
                idx, name, mem_u, mem_t, util, temp = parts[:6]
                lines.append(
                    f"GPU {idx} ({name}): {mem_u:>5s}/{mem_t} MiB | Util: {util:>3s}% | Temp: {temp}°C"
                )
        return lines if lines else ["No NVIDIA GPUs detected."]
    except Exception:
        return ["GPU status unavailable (nvidia-smi not found or local CPU environment)."]


def get_running_osr_processes() -> List[Tuple[str, str]]:
    """Return list of (pid, cmdline) for running train.py or sweep.py jobs."""
    if sys.platform.startswith("win"):
        return []
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], stderr=subprocess.DEVNULL).decode(
            "utf-8", errors="replace"
        )
        procs = []
        for line in out.strip().splitlines():
            line = line.strip()
            if ("scripts/train.py" in line or "scripts/sweep.py" in line) and "grep" not in line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    procs.append((parts[0], parts[1]))
        return procs
    except Exception:
        return []


def list_configs(subdir: str) -> List[str]:
    pattern = os.path.join(ROOT_DIR, "configs", subdir, "*.yaml")
    return [os.path.relpath(p, ROOT_DIR).replace("\\", "/") for p in sorted(glob.glob(pattern))]


def list_experiment_dirs() -> List[str]:
    runs_dir = os.path.join(ROOT_DIR, "runs")
    if not os.path.exists(runs_dir):
        return []
    dirs = []
    for d in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, d)
        if os.path.isdir(full):
            dirs.append(f"runs/{d}")
    return dirs


def find_checkpoints() -> List[str]:
    search_roots = [
        os.path.join(ROOT_DIR, "runs"),
        "/mnt/sbd1_newdisk/project_test_1_22/RARPL_LAST/export",
        "/mnt/sbd1_newdisk/project_test_1_22/RARPL/export",
    ]
    ckpts = []
    for r in search_roots:
        if os.path.exists(r):
            for p in sorted(glob.glob(os.path.join(r, "**", "*.h5"), recursive=True)):
                ckpts.append(p)
    return ckpts


# ==============================================================================
# Curses Interactive Selector
# ==============================================================================
def curses_menu(
    stdscr,
    title: str,
    subtitle: str,
    options: List[str],
    footer: str = "↑/↓: 移动 | Enter: 确认 | q/Esc: 返回",
) -> int:
    """Display an arrow-key navigable menu in curses and return selected index (-1 if cancelled)."""
    curses.curs_set(0)
    current_row = 0
    top_row = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # Header box
        header_title = f"=== ML_Framework_for_OSR 终端控制台 | {title} ==="
        stdscr.addstr(0, 2, header_title[: w - 4], curses.A_BOLD | curses.color_pair(1))

        gpu_lines = get_gpu_status_lines()
        procs = get_running_osr_processes()
        row_y = 1
        for gl in gpu_lines[:2]:
            stdscr.addstr(row_y, 2, f"[硬件] {gl}"[: w - 4], curses.color_pair(2))
            row_y += 1
        stdscr.addstr(
            row_y,
            2,
            f"[后台任务] 当前运行中任务数: {len(procs)}"[: w - 4],
            curses.color_pair(3) if procs else curses.A_DIM,
        )
        row_y += 1
        stdscr.addstr(row_y, 2, "-" * min(w - 4, 86))
        row_y += 1

        if subtitle:
            for s_line in subtitle.splitlines():
                stdscr.addstr(row_y, 2, s_line[: w - 4], curses.A_BOLD)
                row_y += 1
            row_y += 1

        max_visible = max(3, h - row_y - 3)
        if current_row < top_row:
            top_row = current_row
        elif current_row >= top_row + max_visible:
            top_row = current_row - max_visible + 1

        for idx in range(top_row, min(len(options), top_row + max_visible)):
            opt_text = f"  {idx + 1:2d}. {options[idx]}"
            y = row_y + (idx - top_row)
            if idx == current_row:
                stdscr.addstr(y, 2, f"> {idx + 1:2d}. {options[idx]}"[: w - 4], curses.A_REVERSE | curses.A_BOLD)
            else:
                stdscr.addstr(y, 2, opt_text[: w - 4])

        # Footer
        stdscr.addstr(h - 2, 2, "-" * min(w - 4, 86))
        stdscr.addstr(h - 1, 2, footer[: w - 4], curses.color_pair(1))
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            current_row = (current_row - 1) % len(options)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            current_row = (current_row + 1) % len(options)
        elif key in (curses.KEY_ENTER, 10, 13):
            return current_row
        elif key in (ord("q"), ord("Q"), 27):
            return -1
        elif ord("1") <= key <= ord("9"):
            num = key - ord("1")
            if num < len(options):
                return num


def curses_input(stdscr, prompt: str, default_val: str = "") -> str:
    """Prompt user for a string inside curses."""
    curses.echo()
    curses.curs_set(1)
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(2, 2, prompt[: w - 4], curses.A_BOLD)
    stdscr.addstr(3, 2, f"(直接按 Enter 使用默认值: {default_val})"[: w - 4], curses.A_DIM)
    stdscr.addstr(5, 2, "> ")
    stdscr.refresh()
    raw = stdscr.getstr(5, 4, 120).decode("utf-8", errors="replace").strip()
    curses.noecho()
    curses.curs_set(0)
    return raw if raw else default_val


def curses_text_viewer(stdscr, title: str, content: str) -> None:
    """Scrollable text viewer in curses."""
    lines = content.splitlines() or ["(空)"]
    top = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 2, f"=== {title} ==="[: w - 4], curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(1, 2, "-" * min(w - 4, 86))
        max_lines = max(3, h - 4)
        for i in range(top, min(len(lines), top + max_lines)):
            stdscr.addstr(2 + (i - top), 2, lines[i][: w - 4])
        stdscr.addstr(h - 2, 2, "-" * min(w - 4, 86))
        stdscr.addstr(
            h - 1,
            2,
            f"行 {top + 1}-{min(len(lines), top + max_lines)}/{len(lines)} | ↑/↓/PgUp/PgDn: 滚动 | q/Enter: 返回"[
                : w - 4
            ],
            curses.color_pair(1),
        )
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            top = max(0, top - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            top = min(max(0, len(lines) - max_lines), top + 1)
        elif key == curses.KEY_PPAGE:
            top = max(0, top - max_lines)
        elif key == curses.KEY_NPAGE:
            top = min(max(0, len(lines) - max_lines), top + max_lines)
        elif key in (ord("q"), ord("Q"), 27, 10, 13):
            break


def run_command_interactive(stdscr, cmd: List[str], env: Optional[Dict[str, str]] = None, bg_log: Optional[str] = None) -> None:
    """Temporarily exit curses to run a command in foreground or launch in background."""
    curses.endwin()
    run_env = env or os.environ.copy()
    print("\n" + "=" * 80)
    print("执行命令:", " ".join(cmd))
    print("=" * 80)

    if bg_log:
        os.makedirs(os.path.dirname(os.path.abspath(bg_log)), exist_ok=True)
        log_f = open(bg_log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=(not sys.platform.startswith("win")),
        )
        print(f"\n[已启动后台任务] PID = {proc.pid}")
        print(f"[日志文件]       {bg_log}")
        print("你可以随时在主菜单 [5] 中查看实时日志，或直接断开 SSH。")
    else:
        try:
            subprocess.run(cmd, cwd=ROOT_DIR, env=run_env)
        except KeyboardInterrupt:
            print("\n[用户中断了前台任务]")

    input("\n按 Enter 键返回终端 GUI 主界面...")
    stdscr.refresh()


# ==============================================================================
# Workflows
# ==============================================================================
def wizard_single_run(stdscr) -> None:
    ds_list = list_configs("datasets")
    ds_idx = curses_menu(stdscr, "步骤 1/4: 选择数据集配置", "请选择要使用的数据集 YAML 配置：", ds_list)
    if ds_idx < 0:
        return
    ds_cfg = ds_list[ds_idx]

    m_list = list_configs("methods")
    m_idx = curses_menu(stdscr, "步骤 2/4: 选择 OSR 方法", f"已选数据集: {ds_cfg}\n请选择要运行的开集识别方法或消融项：", m_list)
    if m_idx < 0:
        return
    m_cfg = m_list[m_idx]

    sample_opts = [
        "使用配置文件默认值 (COMP=100, WIDEFT=3000)",
        "小样本 Few-Shot: 每类 10 条训练样本 (train_samples_per_class=10)",
        "小样本 Few-Shot: 每类 50 条训练样本 (train_samples_per_class=50)",
        "标准设定: 每类 100 条训练样本 (train_samples_per_class=100)",
        "大样本/重切分设定: 每类 500 条训练样本 (train_samples_per_class=500, pool_and_resplit=true)",
        "自定义每类训练样本数...",
    ]
    s_idx = curses_menu(
        stdscr,
        "步骤 3/4: 选择每类训练样本规模",
        f"数据集: {ds_cfg} | 方法: {m_cfg}",
        sample_opts,
    )
    if s_idx < 0:
        return

    overrides = []
    n_tag = "default"
    if s_idx == 1:
        overrides.append("data.train_samples_per_class=10")
        n_tag = "n10"
    elif s_idx == 2:
        overrides.append("data.train_samples_per_class=50")
        n_tag = "n50"
    elif s_idx == 3:
        overrides.append("data.train_samples_per_class=100")
        n_tag = "n100"
    elif s_idx == 4:
        overrides.extend(["data.train_samples_per_class=500", "data.pool_and_resplit=true"])
        n_tag = "n500"
    elif s_idx == 5:
        custom_n = curses_input(stdscr, "请输入每类训练样本数量 (整数，例如 200):", "100")
        overrides.append(f"data.train_samples_per_class={custom_n}")
        if int(custom_n) > 100 and "comp" in ds_cfg.lower():
            overrides.append("data.pool_and_resplit=true")
        n_tag = f"n{custom_n}"

    split_seed = curses_input(stdscr, "请输入已知/未知类别划分随机种子 (split_seed):", "42")
    overrides.extend([f"data.split_seed={split_seed}", f"seed={split_seed}"])

    epochs_str = curses_input(stdscr, "训练轮数 Epochs (留空使用配置文件默认值):", "")
    if epochs_str:
        overrides.append(f"train.epochs={epochs_str}")

    gpu_opts = [
        "GPU 0 + 前台实时显示进度",
        "GPU 1 + 前台实时显示进度",
        "GPU 0 + 后台挂起运行 (nohup 写入 train.log)",
        "GPU 1 + 后台挂起运行 (nohup 写入 train.log)",
    ]
    g_idx = curses_menu(
        stdscr,
        "步骤 4/4: 选择 GPU 与运行方式",
        f"数据集: {ds_cfg}\n方法:   {m_cfg}\n覆盖项: {' '.join(overrides)}",
        gpu_opts,
    )
    if g_idx < 0:
        return

    gpu_id = "0" if g_idx in (0, 2) else "1"
    is_bg = g_idx in (2, 3)

    ds_short = os.path.splitext(os.path.basename(ds_cfg))[0]
    m_short = os.path.splitext(os.path.basename(m_cfg))[0]
    out_dir = f"runs/{ds_short}/{m_short}_{n_tag}_split{split_seed}"

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--dataset-config",
        ds_cfg,
        "--method-config",
        m_cfg,
        "--output-dir",
        out_dir,
        "--override",
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    bg_log = os.path.join(ROOT_DIR, out_dir, "train.log") if is_bg else None
    run_command_interactive(stdscr, cmd, env=env, bg_log=bg_log)


def wizard_sweep(stdscr) -> None:
    exp_list = list_configs("experiments")
    idx = curses_menu(
        stdscr,
        "双卡 P40 批量实验调度 (Sweep)",
        "选择一个批量实验配置（自动分配到 GPU 0 和 GPU 1 并行队列）：",
        exp_list,
    )
    if idx < 0:
        return
    sweep_cfg = exp_list[idx]

    mode_opts = [
        "预览任务队列 (--dry-run，不实际训练)",
        "前台启动双卡并行队列 (GPU 0 & GPU 1)",
        "后台挂起启动双卡并行队列 (nohup，可安全断开 SSH)",
    ]
    m_idx = curses_menu(stdscr, f"批量调度: {sweep_cfg}", "请选择启动模式：", mode_opts)
    if m_idx < 0:
        return

    cmd = [sys.executable, "scripts/sweep.py", "--sweep-config", sweep_cfg, "--gpus", "0", "1"]
    if m_idx == 0:
        cmd.append("--dry-run")
        run_command_interactive(stdscr, cmd)
    elif m_idx == 1:
        run_command_interactive(stdscr, cmd)
    else:
        exp_name = os.path.splitext(os.path.basename(sweep_cfg))[0]
        bg_log = os.path.join(ROOT_DIR, "runs", exp_name, "sweep_master.log")
        run_command_interactive(stdscr, cmd, bg_log=bg_log)


def wizard_eval_checkpoint(stdscr) -> None:
    ckpts = find_checkpoints()
    if not ckpts:
        curses_text_viewer(stdscr, "未找到权重", "未在 runs/ 或 ../RARPL/export 中找到 .h5 权重文件。")
        return

    c_idx = curses_menu(stdscr, "步骤 1/3: 选择要评测的 .h5 权重文件", "支持新框架权重与旧版 RARPL/WIDEFT/COMP 权重：", ckpts)
    if c_idx < 0:
        return
    ckpt_path = ckpts[c_idx]

    ds_list = list_configs("datasets")
    ds_idx = curses_menu(stdscr, "步骤 2/3: 选择对应的数据集配置", f"权重: {ckpt_path}", ds_list)
    if ds_idx < 0:
        return

    m_list = list_configs("methods")
    m_idx = curses_menu(stdscr, "步骤 3/3: 选择对应的模型/方法配置", f"权重: {ckpt_path}\n数据集: {ds_list[ds_idx]}", m_list)
    if m_idx < 0:
        return

    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = f"runs/eval_ckpt/{ckpt_tag}"
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--dataset-config",
        ds_list[ds_idx],
        "--method-config",
        m_list[m_idx],
        "--output-dir",
        out_dir,
        "--eval-only",
        "--checkpoint",
        ckpt_path,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    run_command_interactive(stdscr, cmd, env=env)


def wizard_view_results(stdscr) -> None:
    exp_dirs = list_experiment_dirs()
    if not exp_dirs:
        curses_text_viewer(stdscr, "实验结果汇总", "runs/ 目录下暂无实验结果。")
        return

    idx = curses_menu(stdscr, "查看与导出实验结果表", "选择要汇总的实验目录（将自动生成 Markdown / CSV / LaTeX 表）：", exp_dirs)
    if idx < 0:
        return
    target_dir = exp_dirs[idx]
    out = subprocess.check_output(
        [sys.executable, "scripts/aggregate.py", "--exp-dir", target_dir],
        cwd=ROOT_DIR,
        stderr=subprocess.STDOUT,
    ).decode("utf-8", errors="replace")

    tex_path = os.path.join(ROOT_DIR, target_dir, "summary.tex")
    if os.path.exists(tex_path):
        with open(tex_path, "r", encoding="utf-8") as f:
            out += "\n=== LaTeX 表格代码 (summary.tex) ===\n" + f.read()

    curses_text_viewer(stdscr, f"实验结果汇总: {target_dir}", out)


def wizard_monitor(stdscr) -> None:
    procs = get_running_osr_processes()
    gpu_lines = get_gpu_status_lines()
    log_files = sorted(
        glob.glob(os.path.join(ROOT_DIR, "runs", "**", "*.log"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )

    lines = ["=== GPU 状态 ==="] + gpu_lines + ["", f"=== 运行中的 OSR 进程 ({len(procs)}) ==="]
    if procs:
        for pid, cmd in procs:
            lines.append(f"  PID {pid}: {cmd}")
    else:
        lines.append("  (当前无运行中的后台训练进程)")

    lines.append("")
    lines.append("=== 最近的训练日志 (最新 5 个) ===")
    for lf in log_files[:5]:
        rel = os.path.relpath(lf, ROOT_DIR)
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lf)))
        lines.append(f"\n--- [{mtime}] {rel} (最后 15 行) ---")
        try:
            with open(lf, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-15:]
                lines.extend([t.rstrip("\r\n") for t in tail])
        except Exception as e:
            lines.append(f"  读取失败: {e}")

    curses_text_viewer(stdscr, "GPU 与后台任务实时监控", "\n".join(lines))


def wizard_inspect_configs(stdscr) -> None:
    all_cfgs = list_configs("datasets") + list_configs("methods") + list_configs("experiments")
    idx = curses_menu(stdscr, "浏览配置文件", "选择要查看的 YAML 配置文件：", all_cfgs)
    if idx < 0:
        return
    cfg_path = os.path.join(ROOT_DIR, all_cfgs[idx])
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()
    curses_text_viewer(stdscr, all_cfgs[idx], content)


def _curses_main(stdscr) -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)

    main_options = [
        "🚀 单次训练与评测向导 (选择数据集 / 方法 / 样本量 10·100·500 / GPU)",
        "⚡ 双卡 P40 批量实验调度 (运行完整对比实验或消融实验队列)",
        "🔍 现有 .h5 权重无泄漏复评 (支持新框架及历史 RARPL/WIDEFT/COMP 权重)",
        "📊 查看与导出实验结果汇总表 (Markdown / CSV / 毕设 LaTeX 表格)",
        "📺 监控 GPU 状态与后台训练日志 (查看 nvidia-smi 与最新 train.log)",
        "⚙️ 浏览 YAML 配置文件 (datasets / methods / experiments)",
        "🚪 退出控制台 (Quit)",
    ]

    while True:
        choice = curses_menu(
            stdscr,
            "主菜单",
            "欢迎使用个体辐射源开集识别 (SEI-OSR) 统一实验框架控制台：",
            main_options,
            footer="↑/↓ 或 数字键 1-7: 选择 | Enter: 进入 | q: 退出",
        )
        if choice in (-1, 6):
            break
        elif choice == 0:
            wizard_single_run(stdscr)
        elif choice == 1:
            wizard_sweep(stdscr)
        elif choice == 2:
            wizard_eval_checkpoint(stdscr)
        elif choice == 3:
            wizard_view_results(stdscr)
        elif choice == 4:
            wizard_monitor(stdscr)
        elif choice == 5:
            wizard_inspect_configs(stdscr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal GUI for ML_Framework_for_OSR.")
    args = parser.parse_args()
    os.chdir(ROOT_DIR)
    curses.wrapper(_curses_main)


if __name__ == "__main__":
    main()
