from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

from osrb.core.registry import METHOD_REGISTRY


def compute_signal_mse(signal: tf.Tensor, reconstruct_signal: tf.Tensor) -> tf.Tensor:
    """Signal reconstruction MSE matching RARPL_LAST implementation."""
    error = signal - reconstruct_signal
    squared_error = tf.abs(error) ** 2
    squared_error_sum = tf.reduce_sum(squared_error)
    total_elements = tf.cast(tf.size(signal), dtype=squared_error_sum.dtype)
    return squared_error_sum / (total_elements + tf.keras.backend.epsilon())


class BaseOSRMethod(ABC):
    """Unified interface for all Open-Set Recognition methods."""

    def __init__(self, method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None):
        self.method_cfg = method_cfg
        self.model_cfg = model_cfg or {}
        self.model: Optional[keras.Model] = None
        self.extractor: Optional[keras.Model] = None
        self.projector: Optional[keras.Model] = None
        self.num_known: int = 0
        self.signal_length: int = 0
        self.class_names: List[str] = []
        self.optimizer: Optional[keras.optimizers.Optimizer] = None

    @abstractmethod
    def build(
        self,
        signal_length: int,
        num_known: int,
        class_names: Optional[List[str]] = None,
        lr: float = 0.001,
    ) -> keras.Model:
        """Build Keras model(s) and optimizer."""

    @abstractmethod
    def train_step(
        self,
        inputs: tf.Tensor,
        targets_soft: tf.Tensor,
        targets_hard: tf.Tensor,
    ) -> Dict[str, float]:
        """Perform a single training step on a batch and return scalar metrics."""

    @abstractmethod
    def eval_step(self, inputs: tf.Tensor, targets_hard: tf.Tensor) -> Dict[str, float]:
        """Perform a single validation step on a batch and return scalar metrics."""

    @abstractmethod
    def fit_calibration(
        self,
        val_known_dict: Dict[str, np.ndarray],
        calib_cfg: Dict[str, Any],
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """Fit open-set rejection thresholds / parameters strictly on validation known classes."""

    @abstractmethod
    def predict_open_set(
        self,
        iq_inputs: np.ndarray,
        batch_size: int = 128,
    ) -> Dict[str, np.ndarray]:
        """
        Run open-set inference on I/Q inputs (N, L, 2).

        Returns dict with keys:
          - 'is_known':     (N,) bool
          - 'pred_classes': (N,) int64 (0..K-1)
          - 'raw_scores':   (N,) float64
          - 'cal_scores':   (N,) float64
          - 'embeddings':   (N, D) float32
          - 'class_scores': (N, K) float64
        """

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been built yet.")
        self.model.save_weights(path)

    def load_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been built yet.")
        self.model.load_weights(path)


def build_method(method_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> BaseOSRMethod:
    """Instantiate an OSR method from configuration."""
    name = method_cfg.get("name", "rarpl")
    cls = METHOD_REGISTRY.get(name)
    return cls(method_cfg=method_cfg, model_cfg=model_cfg)
