from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import KMeans


def compute_auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute AUROC where known=1 and unknown=0 (higher score = more likely known)."""
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def compute_oscr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_preds: np.ndarray,
    known_targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Open-Set Classification Rate (OSCR) area and CCR at specific FPR operating points.

    Parameters
    ----------
    known_scores   : (N_k,) confidence scores for known test samples
    unknown_scores : (N_u,) confidence scores for unknown test samples
    known_preds    : (N_k,) predicted closed-set class indices (0..K-1)
    known_targets  : (N_k,) ground-truth closed-set class indices (0..K-1)
    """
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return {"oscr": float("nan"), "ccr_at_fpr05": float("nan"), "ccr_at_fpr10": float("nan"), "ccr_at_fpr20": float("nan")}

    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    correct_mask = (np.asarray(known_preds) == np.asarray(known_targets))

    n_k = len(known_scores)
    n_u = len(unknown_scores)

    # Sort all unique thresholds descending
    all_thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))[::-1]
    # Vectorized computation using searchsorted on ascending arrays
    uk_sorted = np.sort(unknown_scores)
    kn_correct_sorted = np.sort(known_scores[correct_mask])

    # Sample accepted if score >= tau
    uk_accepted = n_u - np.searchsorted(uk_sorted, all_thresholds, side="left")
    kn_correct_accepted = len(kn_correct_sorted) - np.searchsorted(
        kn_correct_sorted, all_thresholds, side="left"
    )

    fpr = np.concatenate([[0.0], uk_accepted / float(n_u), [1.0]])
    ccr = np.concatenate([[0.0], kn_correct_accepted / float(n_k), [len(kn_correct_sorted) / float(n_k)]])

    # Sort by fpr ascending for trapezoidal integration
    order = np.argsort(fpr, kind="mergesort")
    fpr_sorted = fpr[order]
    ccr_sorted = ccr[order]
    oscr_val = float(np.trapz(ccr_sorted, fpr_sorted))

    def _ccr_at_fpr(target_fpr: float) -> float:
        valid = np.where(fpr_sorted <= target_fpr)[0]
        if len(valid) == 0:
            return 0.0
        return float(np.max(ccr_sorted[valid]))

    return {
        "oscr": oscr_val,
        "ccr_at_fpr05": _ccr_at_fpr(0.05),
        "ccr_at_fpr10": _ccr_at_fpr(0.10),
        "ccr_at_fpr20": _ccr_at_fpr(0.20),
    }


def compute_open_set_metrics(
    num_known: int,
    known_targets: np.ndarray,
    known_preds: np.ndarray,
    known_is_accepted: np.ndarray,
    known_raw_scores: np.ndarray,
    known_cal_scores: np.ndarray,
    unknown_preds: np.ndarray,
    unknown_is_accepted: np.ndarray,
    unknown_raw_scores: np.ndarray,
    unknown_cal_scores: np.ndarray,
    num_unknown_classes: int = 0,
    unknown_true_sublabels: Optional[np.ndarray] = None,
    known_embeddings: Optional[np.ndarray] = None,
    unknown_embeddings: Optional[np.ndarray] = None,
    cluster_unknowns: bool = False,
) -> Dict[str, Any]:
    """
    Compute comprehensive closed-set, threshold-free, and thresholded OSR metrics.
    """
    n_k = len(known_targets)
    n_u = len(unknown_is_accepted)

    known_targets = np.asarray(known_targets, dtype=np.int64)
    known_preds = np.asarray(known_preds, dtype=np.int64)
    known_is_accepted = np.asarray(known_is_accepted, dtype=bool)
    unknown_preds = np.asarray(unknown_preds, dtype=np.int64)
    unknown_is_accepted = np.asarray(unknown_is_accepted, dtype=bool)

    # 1. Closed-set accuracy on known evaluation samples
    closed_acc = float(np.mean(known_preds == known_targets)) if n_k > 0 else float("nan")

    # 2. Thresholded operating-point metrics
    known_detection_rate = float(np.mean(known_is_accepted)) if n_k > 0 else float("nan")
    open_ccr = (
        float(np.mean(known_is_accepted & (known_preds == known_targets)))
        if n_k > 0
        else float("nan")
    )
    unknown_rejection_rate = float(np.mean(~unknown_is_accepted)) if n_u > 0 else float("nan")

    # Identification precision among all samples accepted as known
    total_accepted_known = int(np.sum(known_is_accepted)) + int(np.sum(unknown_is_accepted))
    correct_accepted_known = int(np.sum(known_is_accepted & (known_preds == known_targets)))
    id_precision = (
        float(correct_accepted_known / total_accepted_known) if total_accepted_known > 0 else 0.0
    )

    # (K+1)-class Open-Set Macro-F1 (classes 0..K-1 are known, class K is Unknown)
    unknown_bucket = num_known
    y_true_k1 = np.concatenate([known_targets, np.full(n_u, unknown_bucket, dtype=np.int64)])
    y_pred_known_part = np.where(known_is_accepted, known_preds, unknown_bucket)
    y_pred_unknown_part = np.where(unknown_is_accepted, unknown_preds, unknown_bucket)
    y_pred_k1 = np.concatenate([y_pred_known_part, y_pred_unknown_part])
    open_macro_f1 = float(
        f1_score(y_true_k1, y_pred_k1, labels=list(range(num_known + 1)), average="macro", zero_division=0)
    )
    overall_k1_acc = float(np.mean(y_true_k1 == y_pred_k1))

    # Binary Known vs Unknown detection Precision / Recall / F1 (matching mltools.Evaluation_index)
    tp_k = int(np.sum(known_is_accepted))
    fp_k = int(np.sum(unknown_is_accepted))
    fn_k = n_k - tp_k
    prec_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
    rec_k = tp_k / n_k if n_k > 0 else 0.0
    f1_k = (2 * prec_k * rec_k / (prec_k + rec_k)) if (prec_k + rec_k) > 0 else 0.0

    tp_u = int(np.sum(~unknown_is_accepted))
    fp_u = int(np.sum(~known_is_accepted))
    fn_u = n_u - tp_u
    prec_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else 0.0
    rec_u = tp_u / n_u if n_u > 0 else 0.0
    f1_u = (2 * prec_u * rec_u / (prec_u + rec_u)) if (prec_u + rec_u) > 0 else 0.0

    # 3. Threshold-free academic metrics (AUROC, AUPR, OSCR)
    auroc_raw = compute_auroc(known_raw_scores, unknown_raw_scores)
    auroc_cal = compute_auroc(known_cal_scores, unknown_cal_scores)

    y_bin = np.concatenate([np.ones(n_k), np.zeros(n_u)])
    s_cal = np.concatenate([known_cal_scores, unknown_cal_scores])
    aupr_in = float(average_precision_score(y_bin, s_cal)) if (n_k > 0 and n_u > 0) else float("nan")
    aupr_out = (
        float(average_precision_score(1.0 - y_bin, -s_cal)) if (n_k > 0 and n_u > 0) else float("nan")
    )

    oscr_raw_dict = compute_oscr(known_raw_scores, unknown_raw_scores, known_preds, known_targets)
    oscr_cal_dict = compute_oscr(known_cal_scores, unknown_cal_scores, known_preds, known_targets)

    metrics: Dict[str, Any] = {
        "closed_acc": closed_acc,
        "open_ccr": open_ccr,
        "known_detection_rate": known_detection_rate,
        "unknown_rejection_rate": unknown_rejection_rate,
        "ccr_plus_rejection_sum": open_ccr + unknown_rejection_rate,
        "id_precision": id_precision,
        "open_macro_f1": open_macro_f1,
        "overall_k1_acc": overall_k1_acc,
        "auroc": auroc_cal,
        "auroc_raw": auroc_raw,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
        "oscr": oscr_cal_dict["oscr"],
        "oscr_raw": oscr_raw_dict["oscr"],
        "ccr_at_fpr05": oscr_cal_dict["ccr_at_fpr05"],
        "ccr_at_fpr10": oscr_cal_dict["ccr_at_fpr10"],
        "ccr_at_fpr20": oscr_cal_dict["ccr_at_fpr20"],
        "known_detect_precision": float(prec_k),
        "known_detect_recall": float(rec_k),
        "known_detect_f1": float(f1_k),
        "unknown_detect_precision": float(prec_u),
        "unknown_detect_recall": float(rec_u),
        "unknown_detect_f1": float(f1_u),
        "num_eval_known": int(n_k),
        "num_eval_unknown": int(n_u),
    }

    # Optional KMeans clustering on rejected samples (matching Test_RARPL_WIDEFT.py)
    if (
        cluster_unknowns
        and num_unknown_classes > 0
        and known_embeddings is not None
        and unknown_embeddings is not None
    ):
        rej_kn_emb = known_embeddings[~known_is_accepted]
        rej_uk_emb = unknown_embeddings[~unknown_is_accepted]
        total_rej = len(rej_kn_emb) + len(rej_uk_emb)
        if total_rej >= num_unknown_classes:
            all_rej_emb = np.vstack([p for p in (rej_kn_emb, rej_uk_emb) if len(p) > 0])
            km_labels = KMeans(
                n_clusters=num_unknown_classes, n_init="auto", random_state=42
            ).fit_predict(all_rej_emb)
            uk_km = km_labels[len(rej_kn_emb) :]
            if unknown_true_sublabels is not None and len(unknown_true_sublabels) == n_u:
                rej_true_sub = unknown_true_sublabels[~unknown_is_accepted]
                uk_sub_correct = int(np.sum(uk_km == rej_true_sub))
                metrics["unknown_cluster_acc"] = float(uk_sub_correct / n_u)
                metrics["all_multi_class_acc"] = float(
                    (correct_accepted_known + uk_sub_correct) / (n_k + n_u)
                )

    return metrics
