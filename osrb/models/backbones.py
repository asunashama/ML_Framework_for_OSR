from typing import Tuple
import tensorflow as tf
from tensorflow import keras

from omni.model import (
    Backbone,
    AE_for_ARPL,
    encoderonly_for_ARPL,
    encoderonly_for_OpenMax,
    decoder as omni_decoder,
)
from omni.model.__RSBU_try__ import encoder_for_ARPL


def build_adaptive_decoder(latent_tensor: tf.Tensor, target_length: int) -> tf.Tensor:
    """
    Build signal reconstruction decoder.
    Uses exact omni.model.decoder for length==1024 (100% weight-compatible with RARPL_LAST),
    and adapts time dimension to target_length when target_length != 1024.
    """
    recon_1024 = omni_decoder(latent_tensor)
    if target_length == 1024:
        return recon_1024
    # Adapt (Batch, 1024, 2) -> (Batch, target_length, 2) via bilinear 1D interpolation
    x = tf.expand_dims(recon_1024, axis=2)  # (Batch, 1024, 1, 2)
    x = tf.image.resize(x, size=[target_length, 1], method="bilinear")
    return tf.squeeze(x, axis=2)


def build_ae_for_arpl_adaptive(length: int) -> keras.Model:
    """Build RSBU-CS AutoEncoder for ARPL/RARPL supporting arbitrary signal length."""
    if length == 1024:
        return AE_for_ARPL(length=1024)
    inputs = keras.Input(shape=(length, 2))
    x = encoder_for_ARPL(inputs)
    reconstruct_signal = build_adaptive_decoder(x, target_length=length)
    return keras.Model(inputs=inputs, outputs=(x, reconstruct_signal))


def build_backbone(
    name: str,
    length: int,
    use_decoder: bool = False,
) -> Tuple[keras.Model, bool]:
    """
    Build feature backbone model.

    Returns
    -------
    backbone_model : keras.Model
    has_decoder    : bool (True if backbone_model outputs (representation, reconstruct_signal))
    """
    key = name.lower()
    if key in ("ae_for_arpl", "rsbu_ae"):
        if use_decoder:
            return build_ae_for_arpl_adaptive(length), True
        return encoderonly_for_ARPL(length), False

    if key in ("encoderonly_for_arpl", "rsbu", "rsbu_encoder"):
        if use_decoder:
            return build_ae_for_arpl_adaptive(length), True
        return encoderonly_for_ARPL(length), False

    if key in ("backbone", "resnet1d_sep"):
        base = Backbone(length)
        if not use_decoder:
            return base, False
        inputs = keras.Input(shape=(length, 2))
        rep = base(inputs)
        recon = build_adaptive_decoder(rep, target_length=length)
        return keras.Model(inputs=inputs, outputs=(rep, recon)), True

    if key == "encoderonly_for_openmax":
        return encoderonly_for_OpenMax(length), False

    raise ValueError(
        f"Unknown backbone '{name}'. Supported: ['backbone', 'ae_for_arpl', 'encoderonly_for_arpl', 'encoderonly_for_openmax']"
    )
