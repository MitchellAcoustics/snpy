"""Skew-elliptical linear models."""

from typing import Optional, Union, Dict
import numpy as np
import pandas as pd
from ..core.base import Distribution


class SkewLinearModel:
    """Implementation of selm (skew-elliptical linear model)."""

    def __init__(self):
        self._r = RInterface()
        self._fitted = False

    def fit(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "SN",
        method: str = "MLE",
        **kwargs,
    ) -> None:
        """Fit model using R's selm."""
        pass

    @property
    def coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        self._check_fitted()
        pass

    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        interval_type: str = "confidence",
        level: float = 0.95,
    ) -> np.ndarray:
        """Generate predictions."""
        self._check_fitted()
        pass
