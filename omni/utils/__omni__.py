import argparse
import math
import os
import pickle
import random
import sys
import time
import typing
from functools import singledispatch as dispatch
from typing import TypeAlias, Optional, Callable, Any, TypeVar, Generic, overload

from tensorflow import keras

if typing.TYPE_CHECKING:
    import keras.api.keras as keras

import tensorflow as tf
import numpy as np

import h5py as HDF

Int: TypeAlias = int
Float: TypeAlias = float
Complex: TypeAlias = complex
String: TypeAlias = str
Str: TypeAlias = str
List: TypeAlias = list
Tuple: TypeAlias = tuple
Dict: TypeAlias = dict
array: TypeAlias = np.ndarray
Array: TypeAlias = np.ndarray
tensor: TypeAlias = tf.Tensor
Tensor: TypeAlias = tf.Tensor

__all__ = ['math', 'os', 'pickle', 'sys', 'time', 'dispatch', 'random', 'argparse',
           'tf', 'keras', 'np', 'HDF',
           'Int', 'Float', 'Complex', 'Str', 'String', 'Array', 'Tensor',
           'List', 'Tuple', 'Dict', 'Any', 'Optional', 'Callable', 'TypeVar', 'Generic',
           'array', 'tensor',
           'overload']
