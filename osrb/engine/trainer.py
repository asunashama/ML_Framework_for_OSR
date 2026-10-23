import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf

from omni import utils
from osrb.augment.signal_aug import augment_on_batch, soften_labels
from osrb.core.config import save_yaml
from osrb.core.seed import set_global_seed
from osrb.data.loader import OSRDataBundle, load_osr_dataset
from osrb.data.transforms import complex_to_iq
from osrb.eval.metrics import compute_open_set_metrics
from osrb.eval.plots import (
    plot_open_set_confusion_matrix,
    plot_score_distribution,
    plot_training_curves,
)
from osrb.methods.base import BaseOSRMethod, build_method


class OSRTrainer:
    """
    Unified training, validation-only calibration, and open-set evaluation engine.
    """

    def __init__(self, cfg: Dict[str, Any], output_dir: str):
        self.cfg = cfg
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.seed = int(cfg.get("seed", 42))
        set_global_seed(self.seed)

        self.bundle: Optional[OSRDataBundle] = None
        self.method: Optional[BaseOSRMethod] = None
        self.history: Dict[str, List[float]] = {}

    def setup(self) -> None:
        """Load dataset and build OSR method & model."""
        set_global_seed(self.seed)
        self.bundle = load_osr_dataset(self.cfg.get("data", {}), default_seed=self.seed)
        self.method = build_method(self.cfg.get("method", {}), self.cfg.get("model", {}))

        lr = float(self.cfg.get("train", {}).get("lr", 0.001))
        self.method.build(
            signal_length=self.bundle.signal_length,
            num_known=self.bundle.num_known,
            class_names=self.bundle.known_classes,
            lr=lr,
        )
        save_yaml(self.cfg, os.path.join(self.output_dir, "config.yaml"))

        print("=" * 78)
        print(f"[OSRTrainer] Task: {self.bundle.task_type} | Method: {self.cfg.get('method', {}).get('name', 'rarpl')}")
        print(f"  Output Dir:      {self.output_dir}")
        print(f"  Seed:            {self.seed} | Split Mode: {self.bundle.metadata['split_mode']} (split_seed={self.bundle.metadata['split_seed']})")
        print(f"  Known Classes:   {self.bundle.num_known} {self.bundle.known_classes}")
        print(f"  Unknown Classes: {self.bundle.num_unknown} {self.bundle.unknown_classes}")
        print(f"  Signal Length:   {self.bundle.signal_length} complex points")
        print(
            f"  Samples:         Train={self.bundle.metadata['total_train_samples']} "
            f"({self.bundle.metadata['train_samples_per_class']}/cls), "
            f"Val={self.bundle.metadata['total_val_samples']}, "
            f"EvalKnown={self.bundle.metadata['total_eval_known_samples']}, "
            f"EvalUnknown={self.bundle.metadata['total_eval_unknown_samples']}"
        )
        print("=" * 78)

    def train(self) -> Dict[str, List[float]]:
        """Run training loop with optional signal augmentation and checkpointing."""
        if self.bundle is None or self.method is None:
            self.setup()

        train_cfg = self.cfg.get("train", {})
        aug_cfg = self.cfg.get("augment", {})

        epochs = int(train_cfg.get("epochs", 70))
        batch_size = int(train_cfg.get("batch_size", 64))
        val_batch_size = int(train_cfg.get("val_batch_size", 128))
        steps_per_epoch = train_cfg.get("steps_per_epoch", None)

        aug_enabled = bool(aug_cfg.get("enabled", False))
        k_segments = int(aug_cfg.get("k_segments", 2))
        phase_mode = str(aug_cfg.get("phase_mode", "quadrant"))
        concat_original = bool(aug_cfg.get("concat_original", False))
        ls_alpha = float(aug_cfg.get("label_smoothing", 0.0))

        dst = utils.Dataset(*utils.make(self.bundle.train_known)).batch(batch_size).shuffle()
        if steps_per_epoch is not None:
            dst._total = int(steps_per_epoch)
        dsv = (
            utils.Dataset(*utils.make(self.bundle.val_known, use=complex_to_iq))
            .batch(val_batch_size)
            .shuffle()
        )

        best_val_acc = -1.0
        best_weights = None
        best_path = os.path.join(self.output_dir, "weights.best.h5")
        last_path = os.path.join(self.output_dir, "weights.last.h5")

        self.history = {}

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            epoch_records: Dict[str, List[float]] = {}

            for step_idx in range(len(dst)):
                samples, targets_hard = dst[step_idx]
                if aug_enabled:
                    samples_in = augment_on_batch(
                        samples,
                        num_of_segments=k_segments,
                        phase_mode=phase_mode,
                        concat_original=concat_original,
                    )
                    if concat_original:
                        targets_hard = np.concatenate([targets_hard, targets_hard], axis=0)
                else:
                    samples_in = samples

                inputs_iq = complex_to_iq(samples_in)
                targets_soft = soften_labels(targets_hard, alpha=ls_alpha)

                step_metrics = self.method.train_step(
                    tf.cast(inputs_iq, tf.float32),
                    tf.cast(targets_soft, tf.float32),
                    tf.cast(targets_hard, tf.float32),
                )
                for k, v in step_metrics.items():
                    epoch_records.setdefault(k, []).append(v)

            dst.on_epoch_end()

            for step_idx in range(len(dsv)):
                val_inputs, val_targets = dsv[step_idx]
                val_metrics = self.method.eval_step(
                    tf.cast(val_inputs, tf.float32),
                    tf.cast(val_targets, tf.float32),
                )
                for k, v in val_metrics.items():
                    epoch_records.setdefault(k, []).append(v)

            dsv.on_epoch_end()

            for k, vals in epoch_records.items():
                self.history.setdefault(k, []).append(float(np.mean(vals)))

            val_acc = self.history["val_accuracy"][-1]
            train_acc = self.history.get("accuracy", [0.0])[-1]
            train_loss = self.history.get("loss", [0.0])[-1]
            dt = time.perf_counter() - t0

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_weights = self.method.model.get_weights()
                self.method.save_weights(best_path)

            self.method.save_weights(last_path)

            if epoch == 1 or epoch % max(1, int(train_cfg.get("log_every", 5))) == 0 or epoch == epochs or is_best:
                star = " *BEST*" if is_best else ""
                print(
                    f"Epoch {epoch:3d}/{epochs:3d} | loss={train_loss:.4f} | "
                    f"train_acc={train_acc * 100:6.2f}% | val_acc={val_acc * 100:6.2f}% | "
                    f"{dt:.2f}s{star}"
                )

        if best_weights is not None:
            self.method.model.set_weights(best_weights)

        with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        plot_training_curves(self.history, os.path.join(self.output_dir, "training_curves.png"))
        print(f"\n[OSRTrainer] Training finished. Best validation accuracy: {best_val_acc * 100:.2f}%")
        return self.history

    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calibrate thresholds on validation known set and evaluate on evaluation known + unknown sets.
        """
        if self.bundle is None or self.method is None:
            self.setup()

        if checkpoint_path is not None:
            print(f"[OSRTrainer] Loading weights from: {checkpoint_path}")
            self.method.load_weights(checkpoint_path)

        calib_cfg = self.cfg.get("calib", {})
        eval_cfg = self.cfg.get("eval", {})
        batch_size = int(eval_cfg.get("batch_size", 128))

        # 1. Fit calibration strictly on validation known classes
        calib_info = self.method.fit_calibration(
            self.bundle.val_known, calib_cfg=calib_cfg, batch_size=batch_size
        )
        with open(os.path.join(self.output_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(calib_info, f, indent=2, ensure_ascii=False)

        # 2. Prepare evaluation known & unknown arrays
        kn_iq_list, kn_targets_list = [], []
        for idx, k_name in enumerate(self.bundle.known_classes):
            arr = self.bundle.eval_known[k_name]
            kn_iq_list.append(complex_to_iq(arr))
            kn_targets_list.append(np.full(len(arr), idx, dtype=np.int64))
        kn_iq = np.vstack(kn_iq_list)
        kn_targets = np.concatenate(kn_targets_list)

        uk_iq_list, uk_sublabels_list = [], []
        for u_idx, u_name in enumerate(self.bundle.unknown_classes):
            arr = self.bundle.eval_unknown[u_name]
            uk_iq_list.append(complex_to_iq(arr))
            uk_sublabels_list.append(np.full(len(arr), u_idx, dtype=np.int64))
        uk_iq = np.vstack(uk_iq_list) if uk_iq_list else np.empty((0, self.bundle.signal_length, 2), dtype=np.float32)
        uk_sublabels = np.concatenate(uk_sublabels_list) if uk_sublabels_list else np.empty(0, dtype=np.int64)

        # 3. Predict open-set on known and unknown evaluation samples
        kn_out = self.method.predict_open_set(kn_iq, batch_size=batch_size)
        uk_out = self.method.predict_open_set(uk_iq, batch_size=batch_size)

        cluster_unknowns = bool(eval_cfg.get("cluster_unknowns", False))
        metrics = compute_open_set_metrics(
            num_known=self.bundle.num_known,
            known_targets=kn_targets,
            known_preds=kn_out["pred_classes"],
            known_is_accepted=kn_out["is_known"],
            known_raw_scores=kn_out["raw_scores"],
            known_cal_scores=kn_out["cal_scores"],
            unknown_preds=uk_out["pred_classes"],
            unknown_is_accepted=uk_out["is_known"],
            unknown_raw_scores=uk_out["raw_scores"],
            unknown_cal_scores=uk_out["cal_scores"],
            num_unknown_classes=self.bundle.num_unknown,
            unknown_true_sublabels=uk_sublabels,
            known_embeddings=kn_out["embeddings"],
            unknown_embeddings=uk_out["embeddings"],
            cluster_unknowns=cluster_unknowns,
        )

        # Record sensitivity across fixed k values (computed from validation mu & sigma without test tuning)
        if hasattr(self.method, "calibrator") and getattr(self.method, "calibrator") is not None:
            cal = self.method.calibrator
            orig_k = cal.k_val
            k_sensitivity = {}
            for test_k in eval_cfg.get("k_grid_report", [0.5, 1.0, 1.5, 2.0]):
                cal.set_k(float(test_k))
                k_kn_acc, k_kn_pred, _, _ = cal.predict(kn_out["class_scores"])
                k_uk_acc, _, _, _ = cal.predict(uk_out["class_scores"])
                c_ccr = float(np.mean(k_kn_acc & (k_kn_pred == kn_targets)))
                c_rej = float(np.mean(~k_uk_acc))
                c_det = float(np.mean(k_kn_acc))
                k_sensitivity[f"k={test_k:.2f}"] = {
                    "open_ccr": c_ccr,
                    "unknown_rejection_rate": c_rej,
                    "known_detection_rate": c_det,
                    "sum_ccr_rej": c_ccr + c_rej,
                }
            cal.set_k(orig_k)
            metrics["validation_k_sensitivity"] = k_sensitivity

        metrics["metadata"] = {
            "method": self.cfg.get("method", {}).get("name", "rarpl"),
            "backbone": self.cfg.get("model", {}).get(
                "backbone", self.cfg.get("method", {}).get("backbone", "ae_for_arpl")
            ),
            "seed": self.seed,
            "split_mode": self.bundle.metadata["split_mode"],
            "split_seed": self.bundle.metadata["split_seed"],
            "train_samples_per_class": self.bundle.metadata["train_samples_per_class"],
            "calibration_mode": calib_cfg.get("mode", "fixed_k"),
            "k_val": calib_info.get("k_val", calib_info.get("tau", 1.0)),
        }

        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            os.path.join(self.output_dir, "scores.npz"),
            known_targets=kn_targets,
            known_preds=kn_out["pred_classes"],
            known_is_accepted=kn_out["is_known"],
            known_raw_scores=kn_out["raw_scores"],
            known_cal_scores=kn_out["cal_scores"],
            unknown_sublabels=uk_sublabels,
            unknown_preds=uk_out["pred_classes"],
            unknown_is_accepted=uk_out["is_known"],
            unknown_raw_scores=uk_out["raw_scores"],
            unknown_cal_scores=uk_out["cal_scores"],
        )

        plot_open_set_confusion_matrix(
            num_known=self.bundle.num_known,
            known_targets=kn_targets,
            known_preds=kn_out["pred_classes"],
            known_is_accepted=kn_out["is_known"],
            unknown_preds=uk_out["pred_classes"],
            unknown_is_accepted=uk_out["is_known"],
            save_path=os.path.join(self.output_dir, "confusion_matrix.png"),
        )
        plot_score_distribution(
            known_scores=kn_out["cal_scores"],
            unknown_scores=uk_out["cal_scores"],
            save_path=os.path.join(self.output_dir, "score_distribution.png"),
            threshold=0.0,
        )

        print("-" * 78)
        print(
            f"[Evaluation Summary] Method={metrics['metadata']['method']} | "
            f"SplitSeed={self.bundle.metadata['split_seed']} | Train/Cls={self.bundle.metadata['train_samples_per_class']}"
        )
        print(
            f"  Threshold-Free : ClosedAcc={metrics['closed_acc'] * 100:6.2f}% | "
            f"AUROC={metrics['auroc'] * 100:6.2f}% (raw={metrics['auroc_raw'] * 100:6.2f}%) | "
            f"OSCR={metrics['oscr'] * 100:6.2f}% (raw={metrics['oscr_raw'] * 100:6.2f}%)"
        )
        print(
            f"  Val-Calibrated : OpenCCR={metrics['open_ccr'] * 100:6.2f}% | "
            f"UnknownRej={metrics['unknown_rejection_rate'] * 100:6.2f}% | "
            f"KnownDet={metrics['known_detection_rate'] * 100:6.2f}% | "
            f"MacroF1={metrics['open_macro_f1'] * 100:6.2f}%"
        )
        print("-" * 78)
        return metrics
