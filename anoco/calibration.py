"""Score calibration for ANoCo (Phase-6).

Maps raw anomaly scores to calibrated probabilities using calibration set statistics.
Addresses the "calibration transfer" problem where calib threshold doesn't generalize
to test OK due to distribution shift.

Methods:
    - temperature: score / T (T optimized on calib to minimize NLL)
    - isotonic: non-parametric monotone mapping (sklearn IsotonicRegression)
    - percentile_rank: score → its percentile rank within the calib distribution
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ScoreCalibrator:
    """Calibrate raw anomaly scores to [0, 1] defect probability."""

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: "temperature", "isotonic", or "percentile_rank"
        """
        self.method = method
        self._fitted = False
        self._temperature: float = 1.0
        self._iso_reg = None
        self._calib_scores: Optional[np.ndarray] = None

    def fit(
        self,
        calib_ok_scores: List[float],
        calib_ng_scores: Optional[List[float]] = None,
    ) -> "ScoreCalibrator":
        """Fit the calibrator on calibration data.

        Args:
            calib_ok_scores: scores for calibration OK images (label=0)
            calib_ng_scores: optional scores for calibration NG images (label=1).
                             If provided, used for isotonic regression.
        """
        ok = np.array(calib_ok_scores, dtype=np.float64)
        self._calib_scores = ok

        if self.method == "temperature":
            # Find T that minimizes NLL on calib (OK should have low probability)
            # P(defect|s) = sigmoid(s/T - bias). Optimize T and bias.
            from scipy.optimize import minimize_scalar

            def nll(T):
                if T <= 0:
                    return 1e10
                # For OK: target=0, loss = -log(1 - sigmoid(s/T))
                # Simplified: just minimize mean OK score / T (push OK probs low)
                # Better: use logistic loss
                logits = ok / T
                # NLL for label=0: log(1 + exp(logits))
                loss = np.logaddexp(0, logits).mean()
                if calib_ng_scores:
                    ng = np.array(calib_ng_scores, dtype=np.float64)
                    logits_ng = ng / T
                    # NLL for label=1: log(1 + exp(-logits_ng))
                    loss += np.logaddexp(0, -logits_ng).mean()
                return loss

            result = minimize_scalar(nll, bounds=(0.01, 1000), method="bounded")
            self._temperature = result.x
            self._fitted = True

        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            if calib_ng_scores is None or len(calib_ng_scores) == 0:
                # Without NG in calib, use a heuristic:
                # Map scores so that calib OK maps to [0, 0.5] range
                # Use percentile-based pseudo-labels
                n = len(ok)
                # Create synthetic labels: OK=0, and use OK score distribution
                # to create a "soft" mapping
                sorted_ok = np.sort(ok)
                # Pseudo-labels: rank/(n+1) scaled to [0, 0.3]
                pseudo_labels = np.linspace(0, 0.3, n)
                self._iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                self._iso_reg.fit(sorted_ok, pseudo_labels)
            else:
                ng = np.array(calib_ng_scores, dtype=np.float64)
                scores = np.concatenate([ok, ng])
                labels = np.concatenate([np.zeros(len(ok)), np.ones(len(ng))])
                self._iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                self._iso_reg.fit(scores, labels)
            self._fitted = True

        elif self.method == "percentile_rank":
            # Score → percentile rank within calib OK distribution
            self._fitted = True

        else:
            raise ValueError(f"unknown calibration method: {self.method!r}")

        return self

    def transform(self, scores: List[float]) -> List[float]:
        """Map raw scores to calibrated [0, 1] probabilities."""
        if not self._fitted:
            raise RuntimeError("ScoreCalibrator not fitted")

        scores_arr = np.array(scores, dtype=np.float64)

        if self.method == "temperature":
            # Sigmoid calibration
            logits = scores_arr / self._temperature
            probs = 1.0 / (1.0 + np.exp(-logits))
            return probs.tolist()

        elif self.method == "isotonic":
            return self._iso_reg.predict(scores_arr).tolist()

        elif self.method == "percentile_rank":
            # Percentile of each score within calib distribution
            from scipy.stats import percentileofscore
            return [percentileofscore(self._calib_scores, s) / 100.0 for s in scores]

        raise RuntimeError("unreachable")

    def get_threshold(self, target_fpr: float) -> float:
        """Get the raw score threshold corresponding to target FPR on calib OK.

        Returns the raw score (not calibrated) at which target_fpr fraction
        of calib OK would exceed.
        """
        if self._calib_scores is None:
            raise RuntimeError("not fitted")
        return float(np.percentile(self._calib_scores, 100 * (1 - target_fpr)))

    @property
    def temperature(self) -> float:
        return self._temperature
