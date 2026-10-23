import argparse
import cmath
import math
import os
import pickle
import random
import re
import sys
import time
import typing
import struct
from functools import singledispatch as dispatch

if sys.version_info >= (3, 10):
    from typing import TypeAlias, Optional, Callable, Any, TypeVar, Generic, overload
else:
    from typing import Optional, Callable, Any, TypeVar, Generic, overload
    from typing import Type as TypeAlias  

# from typing import TypeAlias, Optional, Callable, Any, TypeVar, Generic, overload
from tensorflow import keras

if typing.TYPE_CHECKING:
    import keras.api.keras as keras

import tensorflow as tf
import numpy as np

import h5py as HDF5

from scipy import linalg

import matplotlib.pyplot as plt
from io import BytesIO
import pywt
from PyEMD import EMD,CEEMDAN
from vmdpy import VMD
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from scipy.signal import stft

try:
    from libmr import MR
except ImportError:
    from .intellisense import MR

# import comm as comm
# import utils as utils
from . import comm
from . import utils
Int: TypeAlias = int
Float: TypeAlias = float
Complex: TypeAlias = complex
List: TypeAlias = list
Tuple: TypeAlias = tuple
Dict: TypeAlias = dict
Array: TypeAlias = np.ndarray
Tensor: TypeAlias = tf.Tensor

array: TypeAlias = np.ndarray

__all__ = ['comm', 'utils',
           'os', 'sys', 'argparse', 'math', 'cmath', 'pickle', 'random', 'time', 're',
           'dispatch', 'overload', 'struct',
           'Int', 'Float', 'Complex', 'List', 'Tuple', 'Dict', 'Any', 'Optional', 'Callable',
           'TypeVar', 'Generic', 'TypeAlias',
           'tf', 'keras', 'np', 'HDF5', 'MR',
           'array', 'Array', 'Tensor',
           'linalg']

__platform__ = 'GPU'

if __platform__ == 'GPU':
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.set_printoptions(3, suppress=True)
