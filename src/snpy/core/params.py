"""Parameter representations for skew distributions.

This module provides dataclasses for Direct (DP) and Centered (CP) parameterizations
of skew distributions, following the conventions from the R 'sn' package.

References
----------
Azzalini, A. with the collaboration of Capitanio, A. (2014).
The Skew-Normal and Related Families. Cambridge University Press.

Arellano-Valle, R. B. and Azzalini, A. (2013).
The centred parameterization and related quantities of the skew-t distribution.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union, Dict, Optional, Tuple, ClassVar
import numpy as np
import numpy.typing as npt
import warnings

# Type aliases
Array = npt.NDArray[np.float64]
Numeric = Union[float, Array]
OptionalNumeric = Optional[Numeric]

# Constants
SN_SKEWNESS_BOUND: float = 0.995272  # From R sn package
EPSILON: float = 1e-8


class Family(Enum):
    """Supported distribution families."""

    SN = auto()  # Skew-normal
    ESN = auto()  # Extended skew-normal
    ST = auto()  # Skew-t
    SC = auto()  # Skew-Cauchy


def _is_positive_definite(matrix: Array) -> bool:
    """Check if a matrix is positive definite using Cholesky decomposition."""
    if not isinstance(matrix, np.ndarray):
        return False
    try:
        if matrix.ndim == 0:
            return float(matrix) > 0
        elif matrix.ndim == 1:
            return np.all(matrix > 0)
        else:
            np.linalg.cholesky(matrix)
            return True
    except np.linalg.LinAlgError:
        return False


def _validate_scale(scale: Numeric) -> None:
    """Validate scale parameter(s)."""
    if isinstance(scale, (int, float)):
        if scale <= 0:
            raise ValueError("Scale parameter must be positive")
    elif isinstance(scale, np.ndarray):
        if scale.ndim == 1:
            if not np.all(scale > 0):
                raise ValueError("All scale parameters must be positive")
        else:
            if not np.allclose(scale, scale.T):
                raise ValueError("Scale matrix must be symmetric")
            if not _is_positive_definite(scale):
                raise ValueError("Scale matrix must be positive definite")
    else:
        raise TypeError("Scale must be numeric or array")


@dataclass(frozen=True)
class Parameters:
    """Base class for parameter sets."""

    family: ClassVar[Family] = None

    def __post_init__(self):
        """Convert numeric inputs to numpy arrays."""
        for field_name, field_type in self.__annotations__.items():
            if field_name == "family":
                continue
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, (int, float)):
                    object.__setattr__(self, field_name, np.array(value))
                elif isinstance(value, list):
                    object.__setattr__(self, field_name, np.array(value))

    # def __str__(self) -> str:
    #     """Simple string representation."""
    #     return f"{self.__class__.__name__}(dimension={self.dimension})"

    def summary(self, precision: int = 4) -> str:
        """Return a detailed summary of the parameters.

        Parameters
        ----------
        precision : int, optional
            Number of decimal places to show for floating point values.
            Default is 4.

        Returns
        -------
        str
            Multi-line string containing parameter summary.
        """
        lines = []

        # Header with type and family
        dim = "univariate" if self.dimension == 1 else f"{self.dimension}-dimensional"
        lines.append(f"{dim} {self.__class__.__name__}")
        lines.append(
            f"Distribution Family: {self.family.name if self.family else 'Not specified'}"
        )
        lines.append("-" * 40)

        # Parameters
        float_fmt = f"{{:.{precision}f}}"

        for name, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    # Scalar array
                    val_str = float_fmt.format(float(value))
                elif value.ndim == 1:
                    # Vector
                    val_str = np.array2string(
                        value, precision=precision, separator=", ", suppress_small=True
                    )
                else:
                    # Matrix - format with aligned columns
                    val_str = np.array2string(
                        value,
                        precision=precision,
                        separator=", ",
                        suppress_small=True,
                        prefix=" " * (len(name) + 2),  # Align with parameter name
                    )
            else:
                val_str = (
                    float_fmt.format(value) if isinstance(value, float) else str(value)
                )

            lines.append(f"{name}: {val_str}")

        return "\n".join(lines)

    @property
    def parameters(self) -> Dict[str, Numeric]:
        """Get parameters as a dictionary."""
        return {
            field: getattr(self, field)
            for field in self.__annotations__
            if field != "family" and getattr(self, field) is not None
        }

    @property
    def dimension(self) -> int:
        """Get the dimension of the parameter set."""
        raise NotImplementedError

    def is_univariate(self) -> bool:
        """Check if parameter set represents univariate distribution."""
        return self.dimension == 1


@dataclass(frozen=True)
class DirectParameters(Parameters):
    """Direct parameterization (DP) of skew distributions.

    Parameters
    ----------
    xi : float or ndarray
        Location parameter (vector for multivariate case)
    omega : float or ndarray
        Scale parameter (matrix for multivariate case)
    alpha : float or ndarray
        Slant parameter (vector for multivariate case)
    tau : float, optional
        Hidden mean for ESN distribution (default: 0)
    nu : float, optional
        Degrees of freedom for ST distribution
    """

    family: ClassVar[Family] = Family.SN  # Default to SN family
    xi: Numeric
    omega: Numeric
    alpha: Numeric
    tau: OptionalNumeric = 0
    nu: OptionalNumeric = None

    def __post_init__(self):
        """Validate parameters."""
        super().__post_init__()

        # Validate scale parameter
        _validate_scale(self.omega)

        # Validate degrees of freedom
        if self.nu is not None:
            if self.nu <= 0:
                raise ValueError("Degrees of freedom (nu) must be positive")

        # Validate dimensions match for multivariate case
        if self.dimension > 1:
            xi = np.atleast_1d(self.xi)
            alpha = np.atleast_1d(self.alpha)

            if xi.shape != alpha.shape:
                raise ValueError(
                    f"Shape mismatch: xi {xi.shape} vs alpha {alpha.shape}"
                )

            # Validate omega dimensions
            omega = np.atleast_2d(self.omega)
            if omega.shape[0] != len(xi) or omega.shape[1] != len(xi):
                raise ValueError(
                    f"Scale matrix shape {omega.shape} incompatible with "
                    f"parameter dimension {len(xi)}"
                )

    @property
    def dimension(self) -> int:
        """Get the dimension of the parameter set."""
        if np.ndim(self.xi) == 0:
            return 1
        return len(np.atleast_1d(self.xi))


@dataclass(frozen=True)
class CenteredParameters(Parameters):
    """Centered parameterization (CP) of skew distributions.

    Parameters
    ----------
    mean : float or ndarray
        Mean (first cumulant)
    scale : float or ndarray
        Scale parameter (std deviation, matrix for multivariate)
    skewness : float or ndarray
        Skewness (gamma1) (vector for multivariate)
    kurtosis : float or ndarray, optional
        Kurtosis (gamma2) for ST distribution
    """

    family: ClassVar[Family] = Family.SN  # Default to SN family
    mean: Numeric
    scale: Numeric
    skewness: Numeric
    kurtosis: OptionalNumeric = None

    def __post_init__(self):
        """Validate parameters."""
        super().__post_init__()

        # Validate scale
        _validate_scale(self.scale)

        # Validate dimensions match for multivariate case
        if self.dimension > 1:
            mean = np.atleast_1d(self.mean)
            skewness = np.atleast_1d(self.skewness)

            if mean.shape != skewness.shape:
                raise ValueError(
                    f"Shape mismatch: mean {mean.shape} vs skewness {skewness.shape}"
                )

            # Validate scale dimensions
            scale = np.atleast_2d(self.scale)
            if scale.shape[0] != len(mean) or scale.shape[1] != len(mean):
                raise ValueError(
                    f"Scale matrix shape {scale.shape} incompatible with "
                    f"parameter dimension {len(mean)}"
                )

        # Validate skewness bounds for univariate case
        if self.dimension == 1:
            skew = float(self.skewness)
            if abs(skew) >= SN_SKEWNESS_BOUND:
                raise ValueError(
                    f"Skewness parameter must be in (-{SN_SKEWNESS_BOUND}, "
                    f"{SN_SKEWNESS_BOUND})"
                )

    @property
    def dimension(self) -> int:
        """Get the dimension of the parameter set."""
        if np.ndim(self.mean) == 0:  # Scalar or 0-d array
            return 1
        return len(np.atleast_1d(self.mean))
