from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from omni import utils
from osrb.core.registry import METHOD_REGISTRY
from osrb.data.transforms import complex_to_iq
from osrb.models.backbones import build_backbone
from osrb.calib.threshold import PerClassMarginCalibrator
from .base import BaseOSRMethod, compute_signal_mse


@METHOD_REGISTRY.register("softmax")
class SoftmaxMethod(BaseOSRMethod):
    """
    Standard Closed-Set Classifier with Maximum Softmax Probability (MSP) / Probability Margin OSR.
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(method_cfg, model_cfg)
        self.backbone_name: str = str(
            self.model_cfg.get("backbone", self.method_cfg.get("backbone", "backbone"))
        )
        self.use_decoder: bool = bool(self.method_cfg.get("use_decoder", False))
        self.mse_weight: float = float(self.method_cfg.get("mse_weight", 1.0))
        self.score_type: str = str(self.method_cfg.get("score_type", "prob"))  # 'prob' or 'logit'
        self.has_decoder: bool = False
        self.calibrator: Optional[PerClassMarginCalibrator] = None
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

        inputs = keras.Input(shape=(signal_length, 2))
        backbone_model, self.has_decoder = build_backbone(
            self.backbone_name, length=signal_length, use_decoder=self.use_decoder
        )

        if self.has_decoder:
            x, reconstruct_signal = backbone_model(inputs)
        else:
            x = backbone_model(inputs)
            reconstruct_signal = None

        logits = keras.layers.Dense(num_known, name="logit")(x)
        probs = keras.layers.Softmax(name="softmax")(logits)

        self.extractor = keras.Model(inputs, x)
        self.projector = keras.Model(x, (logits, probs))
        if self.has_decoder:
            self.model = keras.Model(inputs, (probs, reconstruct_signal))
        else:
            self.model = keras.Model(inputs, probs)

        self.optimizer = keras.optimizers.Adam(lr, amsgrad=True)
        self.model.compile(self.optimizer)
        self.calibrator = PerClassMarginCalibrator(num_known, self.class_names)
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
                ce_loss = utils.CrossEntropy(probs, targets_soft)
                total_loss = ce_loss + (mse_weight * mse_loss if has_decoder else 0.0)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = utils.CategoricalAccuracy(probs, targets_hard)
            return total_loss, ce_loss, mse_loss, acc

        @tf.function
        def _eval_step_fn(inputs, targets_hard):
            if has_decoder:
                probs, _ = model(inputs, training=False)
            else:
                probs = model(inputs, training=False)
            acc = utils.CategoricalAccuracy(probs, targets_hard)
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

    def _extract_scores(self, iq_inputs: np.ndarray, batch_size: int = 128):
        embeddings = self.extractor.predict(iq_inputs, batch_size=batch_size, verbose=0)
        logits, probs = self.projector.predict(embeddings, batch_size=batch_size, verbose=0)
        scores = logits if self.score_type == "logit" else probs
        return (
            np.asarray(scores, dtype=np.float64),
            np.asarray(embeddings, dtype=np.float32),
            np.asarray(logits, dtype=np.float64),
            np.asarray(probs, dtype=np.float64),
        )

    def fit_calibration(
        self,
        val_known_dict: Dict[str, np.ndarray],
        calib_cfg: Dict[str, Any],
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        all_scores, all_labels = [], []
        for label_idx, name in enumerate(val_known_dict):
            iq = complex_to_iq(val_known_dict[name])
            scores, _, _, _ = self._extract_scores(iq, batch_size=batch_size)
            all_scores.append(scores)
            all_labels.append(np.full(len(iq), label_idx, dtype=np.int64))

        val_scores = np.vstack(all_scores)
        val_labels = np.concatenate(all_labels)

        k_val = float(calib_cfg.get("k_val", 1.0))
        calib_mode = str(calib_cfg.get("mode", "fixed_k"))
        target_tpr = float(calib_cfg.get("target_val_tpr", 0.90))
        return self.calibrator.fit(
            val_scores, val_labels, k_val=k_val, calib_mode=calib_mode, target_val_tpr=target_tpr
        )

    def predict_open_set(
        self,
        iq_inputs: np.ndarray,
        batch_size: int = 128,
    ) -> Dict[str, np.ndarray]:
        scores, embeddings, logits, probs = self._extract_scores(iq_inputs, batch_size=batch_size)
        is_known, pred_classes, _, cal_scores = self.calibrator.predict(scores)
        # For MSP/MLS, standard raw score is max_c(prob) or max_c(logit)
        raw_max = np.max(logits if self.score_type == "logit" else probs, axis=1)
        return {
            "is_known": is_known,
            "pred_classes": pred_classes,
            "raw_scores": raw_max,
            "cal_scores": cal_scores,
            "embeddings": embeddings,
            "class_scores": scores,
        }


@METHOD_REGISTRY.register("mls")
class MLSMethod(SoftmaxMethod):
    """
    Maximum Logit Score (MLS, Vaze et al., ICLR 2022) OSR baseline.
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(method_cfg)
        cfg["score_type"] = "logit"
        super().__init__(cfg, model_cfg)
