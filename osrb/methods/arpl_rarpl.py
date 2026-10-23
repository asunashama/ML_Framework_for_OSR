from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from omni import utils
from osrb.core.registry import METHOD_REGISTRY
from osrb.data.transforms import complex_to_iq
from osrb.models.backbones import build_backbone
from osrb.models.heads import ReciprocalProjector, build_embedding_neck
from osrb.calib.threshold import PerClassMarginCalibrator
from .base import BaseOSRMethod, compute_signal_mse


def _distance_cross_entropy(outputs: tf.Tensor, targets: tf.Tensor, gamma: float = 1.0) -> tf.Tensor:
    a = tf.exp(gamma * outputs)
    b = tf.reduce_sum(tf.exp(gamma * outputs), axis=-1, keepdims=True)
    scores = tf.math.divide_no_nan(a, b)
    return utils.CrossEntropy(scores, targets)


def _regularization_loss(
    outputs: tf.Tensor,
    distance_euclidean: tf.Tensor,
    radius: tf.Tensor,
    targets_hard: tf.Tensor,
    reg_mode: str = "all_classes_hinge",
) -> tf.Tensor:
    """
    Open-space regularization loss:
      - 'all_classes_hinge': mean(max(outputs - radius, 1e-15))  [RARPL / RARPL_LAST design]
      - 'target_hinge':      mean(max(d_euc(x, P_y) - radius, 1e-15))
      - 'target_sq':         mean(square(max(sqrt(d_euc(x, P_y)) - radius, 0))) [Original ARPL paper]
    """
    mode = reg_mode.lower()
    if mode == "all_classes_hinge":
        margin = tf.maximum(outputs - radius, tf.constant(1e-15, dtype=tf.float32))
        return tf.reduce_mean(margin)

    idx = tf.reshape(tf.argmax(targets_hard, axis=-1), (-1, 1))
    target_d_euc = tf.gather(distance_euclidean, idx, axis=1, batch_dims=1)
    if mode == "target_hinge":
        margin = tf.maximum(target_d_euc - radius, tf.constant(1e-15, dtype=tf.float32))
        return tf.reduce_mean(margin)
    if mode == "target_sq":
        dist_l2 = tf.sqrt(tf.maximum(target_d_euc, 1e-12))
        margin = tf.maximum(dist_l2 - radius, 0.0)
        return tf.reduce_mean(tf.square(margin))
    raise ValueError(f"Unsupported reg_mode '{reg_mode}'")


@METHOD_REGISTRY.register("rarpl")
class RARPLMethod(BaseOSRMethod):
    """
    Reconstruction + Adversarial Reciprocal Points Learning (RARPL) and its ablations.

    Configurable switches in method_cfg / model_cfg:
      - backbone: 'ae_for_arpl' | 'encoderonly_for_arpl' | 'backbone'
      - use_decoder: bool (True for full RARPL with signal reconstruction MSE, False for no-decoder / COMP ARPL)
      - use_entropy: bool (True by default; False for no-entropy ablation)
      - use_regularization: bool (True by default; False for no-regularization ablation)
      - only_reconstruction: bool (False by default; True freezes Projector and trains only on MSE)
      - num_features: int (e.g. 10 on WIDEFT, 32 on COMP)
      - lambda_val: float (regularization weight, e.g. 0.5 or 0.1)
      - mse_weight: float (reconstruction loss weight, default 1.0)
      - gamma: float (softmax temperature, default 1.0)
      - distance_mode: 'abs' | 'signed' | 'euclidean'
      - reg_mode: 'all_classes_hinge' | 'target_hinge' | 'target_sq'
      - final_bn: bool (Optional override for embedding neck final BatchNorm)
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(method_cfg, model_cfg)
        self.backbone_name: str = str(
            self.model_cfg.get("backbone", self.method_cfg.get("backbone", "ae_for_arpl"))
        )
        self.use_decoder: bool = bool(
            self.method_cfg.get("use_decoder", self.backbone_name.lower() in ("ae_for_arpl", "rsbu_ae"))
        )
        self.use_entropy: bool = bool(self.method_cfg.get("use_entropy", True))
        self.use_regularization: bool = bool(self.method_cfg.get("use_regularization", True))
        self.only_reconstruction: bool = bool(self.method_cfg.get("only_reconstruction", False))
        self.num_features: int = int(
            self.model_cfg.get("num_features", self.method_cfg.get("num_features", 32))
        )
        self.lambda_val: float = float(self.method_cfg.get("lambda_val", 0.1))
        self.mse_weight: float = float(self.method_cfg.get("mse_weight", 1.0))
        self.gamma: float = float(self.method_cfg.get("gamma", 1.0))
        self.distance_mode: str = str(self.method_cfg.get("distance_mode", "abs"))
        self.reg_mode: str = str(self.method_cfg.get("reg_mode", "all_classes_hinge"))

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
        proj_layer = ReciprocalProjector(
            num_known, self.num_features, distance_mode=self.distance_mode, name="projector"
        )
        if self.only_reconstruction:
            proj_layer.trainable = False

        outputs = proj_layer(embeddings)
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
        use_entropy = self.use_entropy
        use_reg = self.use_regularization
        only_rc = self.only_reconstruction
        gamma = self.gamma
        lambda_val = self.lambda_val
        mse_weight = self.mse_weight
        reg_mode = self.reg_mode
        num_known = self.num_known

        @tf.function
        def _train_step_fn(inputs, targets_soft, targets_hard):
            with tf.GradientTape() as tape:
                if has_decoder:
                    (outputs, d_euc, d_cos, radius), recon = model(inputs, training=True)
                    mse_loss = compute_signal_mse(inputs, recon)
                else:
                    outputs, d_euc, d_cos, radius = model(inputs, training=True)
                    mse_loss = tf.constant(0.0, dtype=tf.float32)

                entropy = _distance_cross_entropy(outputs, targets_soft, gamma)
                reg = _regularization_loss(outputs, d_euc, radius, targets_hard, reg_mode=reg_mode)

                if only_rc:
                    total_loss = mse_loss
                else:
                    total_loss = tf.constant(0.0, dtype=tf.float32)
                    if use_entropy:
                        total_loss = total_loss + entropy
                    if use_reg:
                        total_loss = total_loss + lambda_val * reg
                    if has_decoder:
                        total_loss = total_loss + mse_weight * mse_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds_oh = tf.one_hot(tf.argmax(outputs, axis=-1), num_known)
            acc = utils.CategoricalAccuracy(preds_oh, targets_hard)
            return total_loss, entropy, reg, mse_loss, acc

        @tf.function
        def _eval_step_fn(inputs, targets_hard):
            if has_decoder:
                (outputs, _, _, _), _ = model(inputs, training=False)
            else:
                outputs, _, _, _ = model(inputs, training=False)
            preds_oh = tf.one_hot(tf.argmax(outputs, axis=-1), num_known)
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

    def _extract_scores_and_embeddings(self, iq_inputs: np.ndarray, batch_size: int = 128):
        if len(iq_inputs) == 0:
            return (
                np.empty((0, self.num_known), dtype=np.float32),
                np.empty((0, self.num_features), dtype=np.float32),
            )
        embeddings = self.extractor.predict(iq_inputs, batch_size=batch_size, verbose=0)
        outputs, _, _, _ = self.projector.predict(embeddings, batch_size=batch_size, verbose=0)
        return np.asarray(outputs, dtype=np.float64), np.asarray(embeddings, dtype=np.float32)

    def fit_calibration(
        self,
        val_known_dict: Dict[str, np.ndarray],
        calib_cfg: Dict[str, Any],
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        all_scores = []
        all_labels = []
        for label_idx, name in enumerate(val_known_dict):
            iq = complex_to_iq(val_known_dict[name])
            scores, _ = self._extract_scores_and_embeddings(iq, batch_size=batch_size)
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
        scores, embeddings = self._extract_scores_and_embeddings(iq_inputs, batch_size=batch_size)
        is_known, pred_classes, raw_margins, cal_scores = self.calibrator.predict(scores)
        return {
            "is_known": is_known,
            "pred_classes": pred_classes,
            "raw_scores": raw_margins,
            "cal_scores": cal_scores,
            "embeddings": embeddings,
            "class_scores": scores,
        }


@METHOD_REGISTRY.register("arpl")
class ARPLMethod(RARPLMethod):
    """
    Original Adversarial Reciprocal Points Learning (Chen et al., TPAMI 2021) baseline:
      - Signed distance: d_euc - d_dot
      - Target-class squared Euclidean margin to radius R
      - No reconstruction decoder by default (unless explicitly enabled)
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(method_cfg)
        cfg.setdefault("use_decoder", False)
        cfg.setdefault("distance_mode", "signed")
        cfg.setdefault("reg_mode", "target_sq")
        super().__init__(cfg, model_cfg)


@METHOD_REGISTRY.register("rpl")
class RPLMethod(RARPLMethod):
    """
    Reciprocal Points Learning (Chen et al., ECCV 2020) baseline:
      - Euclidean distance only: d_euc
      - Target-class squared Euclidean margin to radius R
      - No reconstruction decoder by default
    """

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(method_cfg)
        cfg.setdefault("use_decoder", False)
        cfg.setdefault("distance_mode", "euclidean")
        cfg.setdefault("reg_mode", "target_sq")
        super().__init__(cfg, model_cfg)
