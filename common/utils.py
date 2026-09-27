"""
Shared utility functions for reproducibility and
computational efficiency reporting.
"""

import random
import time

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    if tf is not None:
        tf.random.set_seed(seed)


def count_params(model):
    """
    Return total number of parameters in a model.
    """

    return int(model.count_params())


class Timer:
    """
    Context manager for measuring elapsed execution time.

    Example
    -------

    with Timer() as timer:
        model.fit(...)

    print(timer.elapsed)
    """

    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = None

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = (
            time.perf_counter() - self.start
        )