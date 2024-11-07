# Abstract base classes and interfaces
"""Base distribution classes."""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, TypeVar
import numpy as np
from .params import DirectParameters, CenteredParameters
from .r.interface import RInterface


class Distribution(ABC):
    """Abstract base class for all distributions."""

    def __init__(self):
        self._r = RInterface()
        self._params = None

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate current parameters."""
        pass

    @property
    def params(self) -> Union[DirectParameters, CenteredParameters]:
        """Get current parameters."""
        if self._params is None:
            raise ValueError("Parameters not set")
        return self._params


class SkewDistribution(Distribution):
    """Base class for skew-symmetric distributions."""

    def __init__(
        self,
        params: Optional[Union[DirectParameters, CenteredParameters]] = None,
        param_type: str = "dp",
    ):
        super().__init__()
        self.param_type = param_type
        if params is not None:
            self._params = params
            self._validate_parameters()

    def cp2dp(self, cp: CenteredParameters) -> DirectParameters:
        """Convert from CP to DP."""
        # Implement using R's cp2dp
        pass

    def dp2cp(self, dp: DirectParameters) -> CenteredParameters:
        """Convert from DP to CP."""
        # Implement using R's dp2cp
        pass
