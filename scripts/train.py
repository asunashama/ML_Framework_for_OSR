import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from osrb.core.config import resolve_config
from osrb.engine.trainer import OSRTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an OSR model.")
    parser.add_argument("--config", type=str, default=None, help="Path to full experiment config YAML.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Path to dataset config YAML.")
    parser.add_argument("--method-config", type=str, default=None, help="Path to method config YAML.")
    parser.add_argument("--output-dir", type=str, default="runs/default_run", help="Directory for outputs.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run calibration + evaluation.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .h5 weights to evaluate or resume.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. data.train_samples_per_class=100 seed=43",
    )
    args = parser.parse_args()

    cfg = resolve_config(
        config_path=args.config,
        dataset_config_path=args.dataset_config,
        method_config_path=args.method_config,
        overrides=args.override,
    )

    trainer = OSRTrainer(cfg=cfg, output_dir=args.output_dir)
    trainer.setup()

    if not args.eval_only:
        trainer.train()
        trainer.evaluate(checkpoint_path=args.checkpoint)
    else:
        ckpt = args.checkpoint or os.path.join(args.output_dir, "weights.best.h5")
        trainer.evaluate(checkpoint_path=ckpt)


if __name__ == "__main__":
    main()
