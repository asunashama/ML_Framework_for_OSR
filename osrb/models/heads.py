import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_embedding_neck(
    representations: tf.Tensor,
    dim_of_representation: int,
    num_of_features: int,
    final_bn: bool = True,
) -> tf.Tensor:
    """
    Project backbone representations into compact embedding space.
    - final_bn=True matches RARPL_LAST / WIDEFT scripts.
    - final_bn=False matches COMP_*_run_ARPL.py scripts.
    """
    embeddings = keras.layers.Dense(dim_of_representation, use_bias=False)(representations)
    embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    embeddings = keras.layers.ReLU()(embeddings)
    embeddings = keras.layers.Dense(num_of_features)(embeddings)
    if final_bn:
        embeddings = keras.layers.BatchNormalization(momentum=0.99)(embeddings)
    return embeddings


class ReciprocalProjector(keras.layers.Layer):
    """
    Reciprocal Point Projector for RPL, ARPL, and RARPL.

    distance_mode:
      - 'abs':       |d_euc - d_dot|  (RARPL / RARPL_LAST intentional improvement)
      - 'signed':    d_euc - d_dot    (Original ARPL formulation, Chen et al. TPAMI 2021)
      - 'euclidean': d_euc            (RPL formulation)
    """

    def __init__(
        self,
        num_of_classes: int,
        num_of_features: int,
        distance_mode: str = "abs",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_of_classes = num_of_classes
        self._num_of_features = num_of_features
        self._distance_mode = distance_mode.lower()
        self._kernel = None
        self._radius = None

    def build(self, input_shape):
        self._kernel = self.add_weight(
            name="Reciprocal",
            shape=(self._num_of_classes, self._num_of_features),
            initializer="random_normal",
            trainable=True,
        )
        self._radius = self.add_weight(
            name="Radius",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_of_classes": self._num_of_classes,
                "num_of_features": self._num_of_features,
                "distance_mode": self._distance_mode,
            }
        )
        return config

    def call(self, inputs, **kwargs):
        batch_sz = tf.shape(inputs)[0]
        embeddings = tf.reshape(inputs, (batch_sz, 1, self._num_of_features))
        embeddings = tf.tile(embeddings, [1, self._num_of_classes, 1])
        distance_euclidean = (
            tf.reduce_sum(tf.square(embeddings - self._kernel), axis=-1) / self._num_of_features
        )
        distance_cosine = tf.tensordot(inputs, self._kernel, axes=[[1], [1]])
        if self._distance_mode == "abs":
            distance = tf.abs(distance_euclidean - distance_cosine)
        elif self._distance_mode == "signed":
            distance = distance_euclidean - distance_cosine
        elif self._distance_mode == "euclidean":
            distance = distance_euclidean
        else:
            raise ValueError(f"Unsupported distance_mode '{self._distance_mode}'")
        return distance, distance_euclidean, distance_cosine, self._radius

    def get_reciprocals(self) -> np.ndarray:
        return self._kernel.value().numpy()

    def get_radius(self) -> np.ndarray:
        return self._radius.value().numpy()


class PrototypeProjector(keras.layers.Layer):
    """Prototype Distance Projector for GCPL / GPL."""

    def __init__(self, num_of_classes: int, num_of_features: int, **kwargs):
        super().__init__(**kwargs)
        self._num_of_classes = num_of_classes
        self._num_of_features = num_of_features
        self._kernel = None

    def build(self, input_shape):
        self._kernel = self.add_weight(
            name="Prototype",
            shape=(self._num_of_classes, self._num_of_features),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_of_classes": self._num_of_classes,
                "num_of_features": self._num_of_features,
            }
        )
        return config

    def call(self, inputs, **kwargs):
        batch_sz = tf.shape(inputs)[0]
        emb = tf.reshape(inputs, (batch_sz, 1, self._num_of_features))
        emb = tf.tile(emb, [1, self._num_of_classes, 1])
        outputs = tf.reduce_sum(tf.square(emb - self._kernel), axis=-1)
        return outputs

    def get_prototypes(self) -> np.ndarray:
        return self._kernel.value().numpy()
