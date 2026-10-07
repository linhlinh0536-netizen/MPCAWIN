"""Explicit legacy and River KSWIN backends.

Legacy preserves scikit-multiflow 0.5.3's test-before-append ordering, sampling
with replacement and exact KS gate. Each instance owns its random generator.
"""
import numbers
import warnings

import numpy as np
from scipy import stats


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer')
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return int(value)


def validate_seed(seed):
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError('random_state must be an integer or None')
        if not 0 <= seed < 2**32:
            raise ValueError('random_state must be in [0, 2**32)')
        return int(seed)
    return None


class KSWIN:
    backend = 'legacy'

    def __init__(self, alpha=0.005, window_size=100, stat_size=30,
                 data=None, random_state=None):
        if not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must be finite and in (0, 1)')
        self.alpha = float(alpha)
        self.window_size = positive_integer(window_size, 'window_size')
        self.stat_size = positive_integer(stat_size, 'stat_size')
        # Legacy deletes one point before drawing with replacement.
        if self.window_size < self.stat_size + 2:
            raise ValueError('legacy window_size must be >= stat_size + 2')
        self.random_state = validate_seed(random_state)
        self._rng = np.random.RandomState(self.random_state)
        self.reset()
        if data is not None:
            values = np.asarray(data, dtype=float)
            if values.ndim != 1 or len(values) > self.window_size or not np.isfinite(values).all():
                raise ValueError('data must be a finite 1D window of at most window_size points')
            self.window = values.copy()

    def reset(self):
        """Clear state without restarting the instance's random sequence."""
        self.window = np.array([], dtype=float)
        self.change_detected = False
        self.p_value = self.statistic = None
        self.last_reference = self.last_recent = None
        self.last_result = None
        self.n = 0

    def add_element(self, input_value):
        if not isinstance(input_value, numbers.Real) or not np.isfinite(input_value):
            raise ValueError('KSWIN input must be one finite number')
        self.n += 1
        self.change_detected = False
        self.p_value = self.statistic = None
        self.last_reference = self.last_recent = None
        if len(self.window) >= self.window_size:
            self.window = self.window[1:]
            self.last_recent = self.window[-self.stat_size:].copy()
            self.last_reference = self._rng.choice(
                self.window[:-self.stat_size], self.stat_size, replace=True)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Exact calculation unsuccessful.*',
                                        category=RuntimeWarning)
                result = stats.ks_2samp(self.last_reference, self.last_recent, method='exact')
            self.statistic, self.p_value = float(result.statistic), float(result.pvalue)
            self.change_detected = bool(self.p_value <= self.alpha and self.statistic > 0.1)
            if self.change_detected:
                self.window = self.window[-self.stat_size:]
        self.window = np.append(self.window, float(input_value))
        self.last_result = {
            'backend': self.backend, 'tested': self.p_value is not None,
            'statistic': self.statistic, 'p_value': self.p_value,
            'reference': None if self.last_reference is None else self.last_reference.tolist(),
            'recent': None if self.last_recent is None else self.last_recent.tolist(),
            'window_length': len(self.window), 'drift_detected': self.change_detected,
        }

    def detected_change(self):
        return self.change_detected


class RiverKSWINAdapter:
    """River is an explicit experimental backend, not the legacy baseline."""
    backend = 'river'

    def __init__(self, alpha=0.005, window_size=100, stat_size=30, random_state=None):
        if not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must be finite and in (0, 1)')
        self.alpha = float(alpha)
        self.window_size = positive_integer(window_size, 'window_size')
        self.stat_size = positive_integer(stat_size, 'stat_size')
        if self.window_size < 2 * self.stat_size:
            raise ValueError('River window_size must be >= 2 * stat_size')
        self.random_state = validate_seed(random_state)
        self.reset()

    def reset(self):
        from river.drift import KSWIN as RiverKSWIN
        self.detector = RiverKSWIN(alpha=self.alpha, window_size=self.window_size,
                                   stat_size=self.stat_size, seed=self.random_state)
        self.last_result = None

    def add_element(self, value):
        if not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError('KSWIN input must be one finite number')
        tested = len(self.detector.window) + 1 >= self.window_size
        self.detector.update(float(value))
        self.last_result = {
            'backend': self.backend, 'tested': tested, 'statistic': None,
            'p_value': float(self.detector.p_value) if tested else None,
            'reference': None, 'recent': None,
            'window_length': len(self.detector.window),
            'drift_detected': bool(self.detector.drift_detected),
        }

    def detected_change(self):
        return bool(self.detector.drift_detected)
