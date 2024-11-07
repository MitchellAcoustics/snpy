"""Univariate and multivariate skew-normal implementations."""

from typing import Optional, Union, Tuple
import numpy as np
from ..core.base import SkewDistribution
from ..core.params import DirectParameters, CenteredParameters


class SkewNormal(SkewDistribution):
    """Univariate skew-normal distribution."""

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        self._validate_parameters()
        r_x = self._r.numpy2rpy(x)
        if self.param_type == "dp":
            return np.array(self._r.sn.dsn(r_x, dp=self._params.parameters))
        # Implement CP case

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        pass

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """Quantile function."""
        pass

    def random(self, size: int = 1) -> np.ndarray:
        """Generate random samples."""
        pass


class MultivariateSkewNormal(SkewDistribution):
    """Multivariate skew-normal distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional multivariate initialization
