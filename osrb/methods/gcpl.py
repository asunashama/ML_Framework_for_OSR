from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from omni import utils
from osrb.core.registry import METHOD_REGISTRY
from osrb.data.transforms import complex_to_iq
from osrb.models.backbones import build_backbone
from osrb.models.heads import PrototypeProjector, build_embedding_neck
from osrb.calib.threshold import PerClassMarginCalibrator
from .base import BaseOSRMethod, compute_signal_mse


def _prototype_cross_entropy(distances: tf.Tensor, targets: tf.Tensor, gamma: float = 1.0) -> tf.Tensor:
    a = tf.exp(-gamma * distances)
    b = tf.reduce_sum(tf.exp(-gamma * distances), axis=-1, keepdims=True)
    scores = tf.math.divide_no_nan(a, b)
    return utils.CrossEntropy(scores, targets)


def _prototype_regularization(distances: tf.Tensor, targets_hard: tf.Tensor) -> tf.Tensor:
    idx = tf.reshape(tf.argmax(targets_hard, axis=-1), (-1, 1))
    d_target = tf.gather(distances, idx, axis=1, batch_dims=1)
    return tf.reduce_mean(d_target)


@METHOD_REGISTRY.register("gcpl")
class GCPLMethod(BaseOSRMethod):
    """
    Generalized Convolutional Prototype Learning (GCPL / GPL) matching Train_GCPL_WIDEFT.py & Test_GCPL_WIDEFT.py.
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(method_cfg, model_cfg)
        self.backbone_name: str = str(
            self.model_cfg.get("backbone", self.method_cfg.get("backbone", "ae_for_arpl"))
        )
        self.use_decoder: bool = bool(
            self.method_cfg.get("use_decoder", self.backbone_name.lower() in ("ae_for_arpl", "rsbu_ae"))
        )
        self.num_features: int = int(
            self.model_cfg.get("num_features", self.method_cfg.get("num_features", 32))
        )
        self.lambda_val: float = float(self.method_cfg.get("lambda_val", 0.01))
        self.mse_weight: float = float(self.method_cfg.get("mse_weight", 1.0))
        self.gamma: float = float(self.method_cfg.get("gamma", 1.0))
        self.recalibrate_prototypes: bool = bool(
            self.method_cfg.get("recalibrate_prototypes", True)
        )
        default_final_bn = self.backbone_name.lower() != "backbone"
        self.final_bn: bool = bool(
            self.model_cfg.get("final_bn", self.method_cfg.get("final_bn", default_final_bn))
        )
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
            dim_rep = int(backbone_model.output[0].shape[1])
            x, reconstruct_signal = backbone_model(inputs)
        else:
            dim_rep = int(backbone_model.output.shape[1])
            x = backbone_model(inputs)
            reconstruct_signal = None

        embeddings = build_embedding_neck(
            x, dim_of_representation=dim_rep, num_of_features=self.num_features, final_bn=self.final_bn
        )
        outputs = PrototypeProjector(num_known, self.num_features, name="projector")(embeddings)

        self.extractor = keras.Model(inputs, embeddings)
        self.projector = keras.Model(embeddings, outputs)
        if self.has_decoder:
            self.model = keras.Model(inputs, (outputs, reconstruct_signal))
        else:
            self.model = keras.Model(inputs, outputs)

        self.optimizer = keras.optimizers.Adam(lr, amsgrad=True)
        self.model.compile(self.optimizer)
        self.calibrator = PerClassMarginCalibrator(num_known, self.class_names)
        self._compile_tf_steps()
        return self.model

    def _compile_tf_steps(self) -> None:
        model = self.model
        optimizer = self.optimizer
        has_decoder = self.has_decoder
        gamma = self.gamma
        lambda_val = self.lambda_val
        mse_weight = self.mse_weight
        num_known = self.num_known

        @tf.function
        def _train_step_fn(inputs, targets_soft, targets_hard):
            with tf.GradientTape() as tape:
                if has_decoder:
                    distances, recon = model(inputs, training=True)
                    mse_loss = compute_signal_mse(inputs, recon)
                else:
                    distances = model(inputs, training=True)
                    mse_loss = tf.constant(0.0, dtype=tf.float32)

                entropy = _prototype_cross_entropy(distances, targets_soft, gamma)
                reg = _prototype_regularization(distances, targets_hard)
                total_loss = entropy + lambda_val * reg + (mse_weight * mse_loss if has_decoder else 0.0)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds_oh = tf.one_hot(tf.argmin(distances, axis=-1), num_known)
            acc = utils.CategoricalAccuracy(preds_oh, targets_hard)
            return total_loss, entropy, reg, mse_loss, acc

        @tf.function
        def _eval_step_fn(inputs, targets_hard):
            if has_decoder:
                distances, _ = model(inputs, training=False)
            else:
                distances = model(inputs, training=False)
            preds_oh = tf.one_hot(tf.argmin(distances, axis=-1), num_known)
            acc = utils.CategoricalAccuracy(preds_oh, targets_hard)
            return acc

        self._train_fn = _train_step_fn
        self._eval_fn = _eval_step_fn

    def train_step(
        self,
        inputs: tf.Tensor,
        targets_soft: tf.Tensor,
        targets_hard: tf.Tensor,
    ) -> Dict[str, float]:
        total_loss, entropy, reg, mse_loss, acc = self._train_fn(inputs, targets_soft, targets_hard)
        res = {
            "loss": float(total_loss.numpy()),
            "entropy": float(entropy.numpy()),
            "regularization": float(reg.numpy()),
            "accuracy": float(acc.numpy()),
        }
        if self.has_decoder:
            res["mse_loss"] = float(mse_loss.numpy())
        return res

    def eval_step(self, inputs: tf.Tensor, targets_hard: tf.Tensor) -> Dict[str, float]:
        acc = self._eval_fn(inputs, targets_hard)
        return {"val_accuracy": float(acc.numpy())}

    def _extract_neg_distances(self, iq_inputs: np.ndarray, batch_size: int = 128):
        if len(iq_inputs) == 0:
            return (
                np.empty((0, self.num_known), dtype=np.float64),
                np.empty((0, self.num_features), dtype=np.float32),
            )
        embeddings = self.extractor.predict(iq_inputs, batch_size=batch_size, verbose=0)
        distances = self.projector.predict(embeddings, batch_size=batch_size, verbose=0)
        # Negate distances so higher score = closer to class prototype
        return -np.asarray(distances, dtype=np.float64), np.asarray(embeddings, dtype=np.float32)

    def fit_calibration(
        self,
        val_known_dict: Dict[str, np.ndarray],
        calib_cfg: Dict[str, Any],
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        # Optional prototype recalibration to empirical validation class means (matching Test_GCPL_WIDEFT.py)
        if self.recalibrate_prototypes:
            centers = []
            for name in val_known_dict:
                iq = complex_to_iq(val_known_dict[name])
                emb = self.extractor.predict(iq, batch_size=batch_size, verbose=0)
                centers.append(np.mean(emb, axis=0))
            self.projector.set_weights([np.asarray(centers, dtype=np.float32)])

        all_scores = []
        all_labels = []
        for label_idx, name in enumerate(val_known_dict):
            iq = complex_to_iq(val_known_dict[name])
            neg_dists, _ = self._extract_neg_distances(iq, batch_size=batch_size)
            all_scores.append(neg_dists)
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
        neg_dists, embeddings = self._extract_neg_distances(iq_inputs, batch_size=batch_size)
        is_known, pred_classes, raw_margins, cal_scores = self.calibrator.predict(neg_dists)
        return {
            "is_known": is_known,
            "pred_classes": pred_classes,
            "raw_scores": raw_margins,
            "cal_scores": cal_scores,
            "embeddings": embeddings,
            "class_scores": neg_dists,
        }
