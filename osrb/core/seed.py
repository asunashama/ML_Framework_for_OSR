import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42, deterministic_ops: bool = False) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducible splits and training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
