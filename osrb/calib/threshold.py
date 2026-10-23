from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_top2_margins(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted class (argmax) and top1 - top2 margin from class score matrix (N, K).
    Higher score in `scores` means higher affinity to that class.
    """
    scores = np.asarray(scores, dtype=np.float64)
    pred_classes = np.argmax(scores, axis=1).astype(np.int64)
    if scores.shape[1] < 2:
        return pred_classes, scores[:, 0].copy()
    top_two = np.partition(scores, kth=-2, axis=1)[:, -2:]
    margins = np.max(top_two, axis=1) - np.min(top_two, axis=1)
    return pred_classes, margins.astype(np.float64)


class PerClassMarginCalibrator:
    """
    Per-class Gaussian margin threshold calibrator:
        tau_c = mu_c - k * sigma_c
    fitted strictly on validation known classes.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"C{i + 1}" for i in range(num_classes)]
        self.mu: np.ndarray = np.zeros(num_classes, dtype=np.float64)
        self.sigma: np.ndarray = np.ones(num_classes, dtype=np.float64)
        self.tau: np.ndarray = np.zeros(num_classes, dtype=np.float64)
        self.k_val: float = 1.0
        self.stats_per_class: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        k_val: float = 1.0,
        calib_mode: str = "fixed_k",
        target_val_tpr: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Fit per-class margin statistics (mu_c, sigma_c) on correctly classified validation samples.

        calib_mode:
          - 'fixed_k': Uses the specified k_val (default 1.0).
          - 'val_tpr': Selects k on validation set so overall known detection rate >= target_val_tpr.
        """
        pred_classes, margins = compute_top2_margins(val_scores)
        val_labels = np.asarray(val_labels, dtype=np.int64).flatten()

        for c in range(self.num_classes):
            name = self.class_names[c]
            mask_correct = (val_labels == c) & (pred_classes == c)
            cls_margins = margins[mask_correct]
            used_fallback = False
            if len(cls_margins) == 0:
                # Fallback to all validation samples of class c if none were classified as c (e.g. only_rc ablation)
                cls_margins = margins[val_labels == c]
                used_fallback = True
            if len(cls_margins) == 0:
                mu_c, sigma_c, min_c, max_c = 0.0, 1.0, 0.0, 0.0
            else:
                mu_c = float(np.mean(cls_margins))
                sigma_c = float(np.std(cls_margins))
                min_c = float(np.min(cls_margins))
                max_c = float(np.max(cls_margins))

            self.mu[c] = mu_c
            self.sigma[c] = sigma_c
            self.stats_per_class[name] = {
                "mu": mu_c,
                "sigma": sigma_c,
                "min": min_c,
                "max": max_c,
                "num_correct": int(np.sum(mask_correct)),
                "fallback_all_class_samples": used_fallback,
            }

        if calib_mode.lower() == "val_tpr":
            best_k = k_val
            for cand_k in np.linspace(0.0, 3.0, 301):
                cand_tau = self.mu - cand_k * self.sigma
                accepted = margins >= cand_tau[pred_classes]
                tpr = float(np.mean(accepted))
                if tpr >= target_val_tpr:
                    best_k = float(cand_k)
                    break
            self.k_val = best_k
        else:
            self.k_val = float(k_val)

        self.tau = self.mu - self.k_val * self.sigma
        return self.to_dict()

    def set_k(self, k_val: float) -> np.ndarray:
        self.k_val = float(k_val)
        self.tau = self.mu - self.k_val * self.sigma
        return self.tau

    def predict(
        self, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply per-class threshold tau_c to score matrix (N, K).

        Returns
        -------
        is_known          : (N,) bool array
        pred_classes      : (N,) int64 array (0..K-1)
        raw_margins       : (N,) float64 array (top1 - top2)
        calibrated_scores : (N,) float64 array (margin - tau[pred_class])
        """
        pred_classes, margins = compute_top2_margins(scores)
        sample_taus = self.tau[pred_classes]
        is_known = margins >= sample_taus
        calibrated_scores = margins - sample_taus
        return is_known, pred_classes, margins, calibrated_scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_val": self.k_val,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "tau": self.tau.tolist(),
            "tau_by_class": {
                self.class_names[c]: float(self.tau[c]) for c in range(self.num_classes)
            },
            "stats_per_class": self.stats_per_class,
        }
