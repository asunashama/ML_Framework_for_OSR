# ML_Framework_for_OSR

A unified TensorFlow/Keras Open-Set Recognition (OSR) experimental framework for **Specific Emitter Identification (SEI)** and RF signal recognition, with extensible interfaces reserved for future modulation/regime recognition (`amc_osr`).

## 1. Project Structure

```text
ML_Framework_for_OSR/
├── omni/                       # Core RF signal processing, dataset & neural backbone library
├── mltools.py                  # Confusion matrix & plotting helpers
├── osrb/                       # Unified Open-Set Recognition Framework
│   ├── core/                   # Component registries (Task / Backbone / Method), seeding & YAML config resolver
│   ├── data/                   # Pickle dataset loader, random/fixed/explicit K/U splits, sample-count slicing & resplitting
│   ├── augment/                # 4-Quadrant orthogonal phase rotation, conjugate flip, K-segment random permutation & label smoothing
│   ├── models/                 # Backbone wrappers (Backbone, AE_for_ARPL, encoderonly_for_ARPL, encoderonly_for_OpenMax) & OSR heads
│   ├── methods/                # Unified OSRMethod implementations:
│   │                           #   - rarpl (Reconstruction + Adversarial Reciprocal Points Learning & ablations)
│   │                           #   - arpl  (Original ARPL formulation)
│   │                           #   - rpl   (Reciprocal Points Learning)
│   │                           #   - gcpl  (Generalized Convolutional Prototype Learning)
│   │                           #   - openmax (MAV + Weibull tail calibration)
│   │                           #   - softmax (MSP) & mls (Maximum Logit Score)
│   ├── calib/                  # Validation-only per-class margin threshold calibration (tau_c = mu_c - k * sigma_c)
│   ├── eval/                   # Closed-set Acc, AUROC, OSCR, AUPR, Open Macro-F1, Open CCR, Unknown Rejection Rate & plots
│   └── engine/                 # Unified OSRTrainer (training loop, checkpointing, calibration & evaluation)
├── configs/
│   ├── datasets/               # Dataset & sample-scale configurations (10-shot, 100-sample, 500-sample, full)
│   ├── methods/                # Method & ablation configurations
│   └── experiments/            # Multi-GPU sweep configurations
├── osr_gui.sh                  # One-key launcher for Interactive Shell Terminal GUI (TUI)
└── scripts/
    ├── tui.py                  # Interactive Curses Terminal GUI (Wizard, GPU Monitor, Results Viewer)
    ├── train.py                # Single training / evaluation entry point
    ├── sweep.py                # Multi-GPU (dual Tesla P40) parallel queue scheduler
    └── aggregate.py            # Aggregate runs into Markdown, CSV, and LaTeX (mean ± std) tables
```

## 2. Interactive Shell Terminal GUI (Recommended)

Launch the interactive terminal interface directly in your SSH shell:

```bash
cd /mnt/sbd1_newdisk/project_test_1_22/osr_bench
./osr_gui.sh
# or:
conda run --no-capture-output -n Taopy310 python scripts/tui.py
```

The TUI provides:
1. **Live GPU Status Header**: Real-time memory, utilization, and temperature for both Tesla P40 GPUs (`GPU 0` & `GPU 1`).
2. **Single Experiment Wizard**: Interactively select dataset, OSR method, training samples per class (`10` / `50` / `100` / `500` / `Full`), epoch count, seed, and GPU, then run in foreground or background (`nohup`).
3. **Multi-GPU Sweep Wizard**: Launch parallel benchmark or ablation queues across `GPU 0,1` or execute with `--dry-run`.
4. **Evaluate Checkpoint (`.h5`)**: Pick any existing `.h5` weight file and evaluate it with zero test-set leakage.
5. **Results & Summary Tables**: Browse completed experiments (`summary.md`), inspect per-run metrics, or trigger `aggregate.py` with one key (`a`).
6. **Live Log & Background Task Monitor**: Monitor running processes and tail `.log` / `train.log` files in real time.

## 3. CLI Quick Start

### Single Training & Evaluation Run
```bash
python scripts/train.py \
  --dataset-config configs/datasets/comp_30classes_random.yaml \
  --method-config configs/methods/rarpl.yaml \
  --output-dir runs/debug_rarpl \
  --override data.split_seed=42 data.train_samples_per_class=100
```

### Evaluate an Existing Checkpoint (No Test-Set Leakage)
```bash
python scripts/train.py \
  --dataset-config configs/datasets/wideft_apple_os.yaml \
  --method-config configs/methods/rarpl.yaml \
  --output-dir runs/eval_existing_wideft_rarpl \
  --eval-only \
  --checkpoint /mnt/sbd1_newdisk/project_test_1_22/RARPL_LAST/export/model.best_WIDEFT_RARPL.h5
```

### Flexible Sample-Size Settings (Few-Shot / Standard / Full)
- **Few-Shot (e.g., 10 samples/class)**:
  `--override data.train_samples_per_class=10`
- **Standard (100 samples/class)**:
  `--override data.train_samples_per_class=100`
- **Large-Sample / Resplit (e.g., 500 samples/class on COMP 30-class dataset)**:
  `--override data.train_samples_per_class=500 data.pool_and_resplit=true`

### Multi-GPU Parallel Sweep & Table Aggregation
```bash
python scripts/sweep.py --sweep-config configs/experiments/comp_main_benchmark.yaml --gpus 0 1
python scripts/aggregate.py --exp-dir runs/comp_main_benchmark
```

## 3. Copyright & License

All rights reserved (No License).
