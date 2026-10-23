from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import weibull_min
import tensorflow as tf
from tensorflow import keras

try:
    from libmr import MR as LibMR
except ImportError:
    LibMR = None

from omni import utils
from osrb.core.registry import METHOD_REGISTRY
from osrb.data.transforms import complex_to_iq
from osrb.models.backbones import build_adaptive_decoder, build_backbone
from .base import BaseOSRMethod, compute_signal_mse


class _WeibullTailModel:
    """Wrapper using libmr.MR when installed, with automatic scipy.stats.weibull_min fallback."""

    def __init__(self):
        self._mr = LibMR() if LibMR is not None else None
        self._scipy_params = None

    def fit_high(self, tail_data: np.ndarray) -> None:
        tail_data = np.asarray(tail_data, dtype=np.float64).ravel()
        if len(tail_data) == 0:
            return
        if self._mr is not None:
            try:
                self._mr.fit_high(tail_data, len(tail_data))
                return
            except Exception:
                self._mr = None
        c, loc, scale = weibull_min.fit(tail_data, floc=0.0)
        self._scipy_params = (c, loc, scale)

    def w_score(self, dist: float) -> float:
        if self._mr is not None:
            return float(self._mr.w_score(float(dist)))
        if self._scipy_params is not None:
            c, loc, scale = self._scipy_params
            return float(weibull_min.cdf(float(dist), c, loc=loc, scale=scale))
        return 0.0


@METHOD_REGISTRY.register("openmax")
class OpenMaxMethod(BaseOSRMethod):
    """
    OpenMax (Bendale & Boult, CVPR 2016) with optional signal reconstruction decoder,
    matching Train_OpenMax_WIDEFT.py and Test_OpenMax_WIDEFT.py.
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(method_cfg, model_cfg)
        self.backbone_name: str = str(
            self.model_cfg.get("backbone", self.method_cfg.get("backbone", "encoderonly_for_openmax"))
        )
        self.use_decoder: bool = bool(self.method_cfg.get("use_decoder", True))
        self.mse_weight: float = float(self.method_cfg.get("mse_weight", 1.0))
        self.tail_size: int = int(self.method_cfg.get("tail_size", 50))
        self.alpha: Optional[int] = self.method_cfg.get("alpha", None)
        self.tau: float = float(self.method_cfg.get("tau", 0.90))
        self.has_decoder: bool = False
        self._mu: List[np.ndarray] = []
        self._weibull: List[_WeibullTailModel] = []
        self._feature_model: Optional[keras.Model] = None
        self._train_fn = None
        self._eval_fn = None

    def build(
        self,
        signal_length: int,
        num_known: int,
        class_names: Optional[List[str]] = None,
        lr: float = 0.001,
    ) -> keras.Model:
        self.signal_length = signal_length
        self.num_known = num_known
        self.class_names = class_names or [f"C{i + 1}" for i in range(num_known)]
        if self.alpha is None:
            self.alpha = num_known

        inputs = keras.Input(shape=(signal_length, 2))
        if self.backbone_name.lower() == "encoderonly_for_openmax":
            enc, _ = build_backbone("encoderonly_for_openmax", length=signal_length, use_decoder=False)
            seq_feat = enc(inputs)
            gap = keras.layers.GlobalAveragePooling1D(name="gap")(seq_feat)
        else:
            enc, _ = build_backbone(self.backbone_name, length=signal_length, use_decoder=False)
            gap = keras.layers.Lambda(lambda t: t, name="gap")(enc(inputs))

        logit = keras.layers.Dense(num_known, name="logit")(gap)
        outputs = keras.layers.Softmax(name="softmax")(logit)

        if self.use_decoder:
            self.has_decoder = True
            restruct_signal = build_adaptive_decoder(logit, target_length=signal_length)
            self.model = keras.Model(inputs=inputs, outputs=(outputs, restruct_signal))
        else:
            self.has_decoder = False
            self.model = keras.Model(inputs=inputs, outputs=outputs)

        self._feature_model = keras.Model(inputs=inputs, outputs=[gap, logit, outputs])
        self.extractor = keras.Model(inputs=inputs, outputs=gap)
        self.optimizer = keras.optimizers.Adam(lr, amsgrad=True)
        self.model.compile(self.optimizer)
        self._compile_tf_steps()
        return self.model

    def _compile_tf_steps(self) -> None:
        model = self.model
        optimizer = self.optimizer
        has_decoder = self.has_decoder
        mse_weight = self.mse_weight

        @tf.function
        def _train_step_fn(inputs, targets_soft, targets_hard):
            with tf.GradientTape() as tape:
                if has_decoder:
                    probs, recon = model(inputs, training=True)
                    mse_loss = compute_signal_mse(inputs, recon)
                else:
                    probs = model(inputs, training=True)
                    mse_loss = tf.constant(0.0, dtype=tf.float32)
                ce_loss = utils.CrossEntropy(targets_soft, probs)
                total_loss = ce_loss + (mse_weight * mse_loss if has_decoder else 0.0)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = utils.CategoricalAccuracy(targets_hard, probs)
            return total_loss, ce_loss, mse_loss, acc

        @tf.function
        def _eval_step_fn(inputs, targets_hard):
            if has_decoder:
                probs, _ = model(inputs, training=False)
            else:
                probs = model(inputs, training=False)
            acc = utils.CategoricalAccuracy(targets_hard, probs)
            return acc

        self._train_fn = _train_step_fn
        self._eval_fn = _eval_step_fn

    def train_step(
        self,
        inputs: tf.Tensor,
        targets_soft: tf.Tensor,
        targets_hard: tf.Tensor,
    ) -> Dict[str, float]:
        total_loss, ce_loss, mse_loss, acc = self._train_fn(inputs, targets_soft, targets_hard)
        res = {
            "loss": float(total_loss.numpy()),
            "entropy": float(ce_loss.numpy()),
            "accuracy": float(acc.numpy()),
        }
        if self.has_decoder:
            res["mse_loss"] = float(mse_loss.numpy())
        return res

    def eval_step(self, inputs: tf.Tensor, targets_hard: tf.Tensor) -> Dict[str, float]:
        acc = self._eval_fn(inputs, targets_hard)
        return {"val_accuracy": float(acc.numpy())}

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 1 and b.ndim == 1:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return 1.0 - float(np.dot(a, b) / denom)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b_norm = np.linalg.norm(b) + 1e-12
        sim = np.dot(a, b).reshape((-1, 1)) / (a_norm * b_norm)
        return (1.0 - sim).ravel()

    def fit_calibration(
        self,
        val_known_dict: Dict[str, np.ndarray],
        calib_cfg: Dict[str, Any],
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        if "tau" in calib_cfg:
            self.tau = float(calib_cfg["tau"])
        if "tail_size" in calib_cfg:
            self.tail_size = int(calib_cfg["tail_size"])

        all_emb, all_logits, all_preds, all_targets = [], [], [], []
        for label_idx, name in enumerate(val_known_dict):
            iq = complex_to_iq(val_known_dict[name])
            emb, logits, probs = self._feature_model.predict(iq, batch_size=batch_size, verbose=0)
            all_emb.append(emb)
            all_logits.append(logits)
            all_preds.append(np.argmax(probs, axis=-1))
            all_targets.append(np.full(len(iq), label_idx, dtype=np.int64))

        embeddings = np.vstack(all_emb)
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        correct_mask = preds == targets
        emb_corr = embeddings[correct_mask]
        tgt_corr = targets[correct_mask]

        self._mu = []
        self._weibull = []
        tail_counts = {}
        for k in range(self.num_known):
            cls_emb = emb_corr[tgt_corr == k]
            if len(cls_emb) == 0:
                cls_emb = embeddings[targets == k]
            mu_k = np.mean(cls_emb, axis=0) if len(cls_emb) > 0 else np.full(embeddings.shape[1], np.nan)
            self._mu.append(mu_k)
            wb = _WeibullTailModel()
            if len(cls_emb) > 0 and not np.isnan(mu_k).any():
                dists = np.sort(self._cosine_distance(cls_emb, mu_k))[-self.tail_size :]
                wb.fit_high(dists)
                tail_counts[self.class_names[k]] = int(len(dists))
            else:
                tail_counts[self.class_names[k]] = 0
            self._weibull.append(wb)

        # Optionally calibrate tau on validation set to achieve target_val_tpr
        if str(calib_cfg.get("mode", "")).lower() == "val_tpr":
            target_tpr = float(calib_cfg.get("target_val_tpr", 0.90))
            val_logits = np.vstack(all_logits)
            val_max_probs = []
            for i in range(len(embeddings)):
                om_prob = self._calibrate_single(embeddings[i], val_logits[i])
                val_max_probs.append(float(np.max(om_prob[:-1])))
            self.tau = float(np.quantile(val_max_probs, 1.0 - target_tpr))

        return {
            "tau": self.tau,
            "tail_size": self.tail_size,
            "alpha": self.alpha,
            "tail_counts": tail_counts,
        }

    def _calibrate_single(self, embedding: np.ndarray, logit: np.ndarray) -> np.ndarray:
        w = np.ones(len(logit), dtype=np.float64)
        alpha = min(self.alpha or len(logit), len(logit))
        rank = np.argsort(-logit)[:alpha]
        for k in range(alpha):
            cls_idx = int(rank[k])
            mu_k = self._mu[cls_idx]
            if np.isnan(mu_k).any():
                continue
            d = float(self._cosine_distance(embedding, mu_k))
            if not np.isnan(d):
                w[cls_idx] = 1.0 - ((alpha - k) / float(alpha)) * self._weibull[cls_idx].w_score(d)
        rev_logit = logit * w
        unknown_logit = float(np.sum(logit * (1.0 - w)))
        om_logit = np.append(rev_logit, unknown_logit)
        shifted = om_logit - np.max(om_logit)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict_open_set(
        self,
        iq_inputs: np.ndarray,
        batch_size: int = 128,
    ) -> Dict[str, np.ndarray]:
        embeddings, logits, _ = self._feature_model.predict(iq_inputs, batch_size=batch_size, verbose=0)
        n = len(iq_inputs)
        is_known = np.zeros(n, dtype=bool)
        pred_classes = np.zeros(n, dtype=np.int64)
        max_known_probs = np.zeros(n, dtype=np.float64)
        known_prob_matrix = np.zeros((n, self.num_known), dtype=np.float64)

        for i in range(n):
            om_prob = self._calibrate_single(embeddings[i], logits[i])
            kn_probs = om_prob[:-1]
            known_prob_matrix[i] = kn_probs
            c = int(np.argmax(kn_probs))
            max_p = float(kn_probs[c])
            pred_classes[i] = c
            max_known_probs[i] = max_p
            is_known[i] = max_p >= self.tau

        cal_scores = max_known_probs - self.tau
        return {
            "is_known": is_known,
            "pred_classes": pred_classes,
            "raw_scores": max_known_probs,
            "cal_scores": cal_scores,
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "class_scores": known_prob_matrix,
        }
