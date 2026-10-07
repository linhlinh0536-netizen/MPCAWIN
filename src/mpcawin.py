"""MPCAWIN: PCA angle -> statistical anomaly window -> binary drift detector.

Defaults preserve the published implementation's reference scaling, rounded
angles/thresholds, and legacy KSWIN configuration. Experimental variants are
explicit options. See KE_HOACH_SUA_THEO_BAI_BAO.md for provenance and limitations.
"""
import copy
import math
import numbers
import queue

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kswin_compat import KSWIN, positive_integer, validate_seed


class MPCAWIN:
    _DETECT_STRATEGIES = ('alert_rate', 'detector')

    def __init__(self, min_window_size=10, max_window_size=10, coef_std=2,
                 detect_strategy='detector', alert_window_size=20, alpha=0.5,
                 detector=None, random_seed=20, scaling='reference',
                 rounding='legacy', pca_solver='auto'):
        self.min_window_size = positive_integer(min_window_size, 'min_window_size')
        self.max_window_size = positive_integer(max_window_size, 'max_window_size')
        if self.max_window_size < self.min_window_size:
            raise ValueError('max_window_size must be >= min_window_size')
        if not isinstance(coef_std, numbers.Real) or not np.isfinite(coef_std) or coef_std < 0:
            raise ValueError('coef_std must be finite and nonnegative')
        if detect_strategy not in self._DETECT_STRATEGIES:
            raise ValueError(f'detect_strategy must be one of {self._DETECT_STRATEGIES}')
        if scaling not in ('reference', 'batch'):
            raise ValueError('scaling must be reference or batch')
        if rounding not in ('legacy', 'full'):
            raise ValueError('rounding must be legacy or full')
        if pca_solver not in ('auto', 'full', 'covariance_eigh'):
            raise ValueError('unsupported pca_solver')
        self.random_seed = validate_seed(random_seed)
        self.coef_std = float(coef_std)
        self.detect_strategy = detect_strategy
        self.scaling, self.rounding, self.pca_solver = scaling, rounding, pca_solver
        self.alert_window_size = positive_integer(alert_window_size, 'alert_window_size')
        if not isinstance(alpha, numbers.Real) or not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alert_rate alpha must be finite and in (0, 1]')
        self.alpha = float(alpha)
        self.detector = detector if detector is not None else KSWIN(
            alpha=0.009, window_size=20, stat_size=5, random_state=self.random_seed)
        if detect_strategy == 'detector':
            for method in ('reset', 'add_element', 'detected_change'):
                if not callable(getattr(self.detector, method, None)):
                    raise TypeError(f'detector must implement {method}()')
        self.concept_id = -1
        self.reset()

    def reset(self):
        self.window = queue.Queue(self.max_window_size)
        self.alert_window = queue.Queue(self.alert_window_size)
        if self.detect_strategy == 'detector':
            self.detector.reset()
        self.threshold = None
        self.scaler = self.pca = None
        self.actual_transform_concept = self.actual_concept_eigenvalues = None
        self.feature_columns = None
        self.last_angle = self.last_noise_angle = self.last_result = None

    @staticmethod
    def _batch(input_value):
        frame = pd.DataFrame(input_value)
        if frame.ndim != 2 or frame.shape[1] < 1 or frame.shape[0] < 2:
            raise ValueError('batch must have at least 2 samples and 1 feature')
        if frame.shape[0] < frame.shape[1]:
            raise ValueError('batch needs at least as many samples as features for a full eigenvalue vector')
        if frame.columns.has_duplicates:
            raise ValueError('feature names must be unique')
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError('all features must be numeric')
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('batch contains NaN or infinity')
        if np.all(values == values[0]):
            raise ValueError('batch has zero variance')
        return frame

    def _new_pca(self):
        return PCA(copy=True, svd_solver=self.pca_solver, random_state=self.random_seed)

    @staticmethod
    def _validate_spectrum(values):
        if not np.isfinite(values).all() or not np.isfinite(np.linalg.norm(values)) or np.linalg.norm(values) == 0:
            raise ValueError('PCA spectrum is undefined: batch has zero variance or exceeds numeric range')

    def set_concept(self, input_value):
        """Fit first; commit only when the new concept is valid."""
        frame = self._batch(input_value)
        scaler = StandardScaler(copy=True)
        standardized = scaler.fit_transform(frame)
        if not np.any(np.var(standardized, axis=0) > 0):
            raise ValueError('cannot set a concept from a zero-variance batch')
        pca = self._new_pca().fit(standardized)
        spectrum = pca.explained_variance_.copy()
        self._validate_spectrum(spectrum)
        transformed = pca.transform(standardized)
        self.reset()
        self.scaler, self.pca = scaler, pca
        self.actual_transform_concept, self.actual_concept_eigenvalues = transformed, spectrum
        self.feature_columns = frame.columns.copy()
        self.concept_id += 1

    def _rounded(self, value):
        return round(float(value), 2) if self.rounding == 'legacy' else float(value)

    def _update_threshold(self):
        values = list(self.window.queue)
        self.threshold = self._rounded(np.mean(values) + self.coef_std * np.std(values))

    @staticmethod
    def _push(window, value):
        if window.full():
            window.get_nowait()
        window.put_nowait(value)

    def add_element_and_detected_change(self, input_value):
        """Consume a numeric batch; return bool and retain decision in last_result.

        A completely constant batch is rejected before state mutation. The first
        batch initializes the concept; the following min_window_size comparisons
        warm up SAD. Labels and source boundaries are never inputs to this method.
        """
        frame = self._batch(input_value)
        before = self.concept_id
        if self.actual_concept_eigenvalues is None:
            self.set_concept(frame)
            self.last_result = dict(angle=None, threshold_used=None, alert=None,
                                    noise_angle=None, ks=None, drift_detected=False,
                                    phase='initialize', concept_before=before,
                                    concept_after=self.concept_id, pca_solver=getattr(self.pca, '_fit_svd_solver', None))
            return False
        if not frame.columns.equals(self.feature_columns):
            raise ValueError('feature names and order must match the current concept')
        standardized = (self.scaler.transform(frame) if self.scaling == 'reference'
                        else StandardScaler().fit_transform(frame))
        if not np.any(np.var(standardized, axis=0) > 0):
            raise ValueError('cannot compare a zero-variance batch')
        batch_pca = self._new_pca().fit(standardized)
        values = batch_pca.explained_variance_
        self._validate_spectrum(values)
        reference = self.actual_concept_eigenvalues
        if values.shape != reference.shape:
            raise ValueError('batch PCA spectrum size differs from the current concept')
        reference_norm, values_norm = np.linalg.norm(reference), np.linalg.norm(values)
        if values_norm > 1 and reference_norm > np.finfo(float).max / values_norm:
            cosine = np.inner(reference / reference_norm, values / values_norm)
        else:
            cosine = np.inner(reference, values) / (reference_norm * values_norm)
        angle = self._rounded(math.degrees(math.acos(float(np.clip(cosine, -1, 1)))))
        threshold = self.threshold
        alert = None if threshold is None else int(angle > threshold)
        ks = None
        detected = False
        if alert is None:
            self._push(self.window, angle)
            if self.window.qsize() >= self.min_window_size:
                self._update_threshold()
        else:
            if self.detect_strategy == 'detector':
                self.detector.add_element(alert)
                detected = bool(self.detector.detected_change())
                ks = copy.deepcopy(getattr(self.detector, 'last_result', None))
            else:
                self._push(self.alert_window, alert)
                detected = self.alert_window.full() and sum(self.alert_window.queue) >= math.ceil(self.alert_window_size * self.alpha)
            if detected:
                self.set_concept(frame)
            elif alert == 0:
                self._push(self.window, angle)
                self._update_threshold()
        self.last_angle = angle
        self.last_noise_angle = angle if alert == 1 else None
        self.last_result = dict(angle=angle, threshold_used=threshold, alert=alert,
                                noise_angle=self.last_noise_angle, ks=ks,
                                drift_detected=bool(detected), phase='warmup' if alert is None else 'evaluate',
                                concept_before=before, concept_after=self.concept_id,
                                pca_solver=getattr(batch_pca, '_fit_svd_solver', None))
        return bool(detected)
