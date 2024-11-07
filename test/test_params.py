"""Tests for parameter validation and handling."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from snpy.core.params import (
    DirectParameters,
    CenteredParameters,
    SN_SKEWNESS_BOUND,
    _is_positive_definite,
)


# Simple valid parameter sets for testing
@pytest.fixture
def valid_dp_univariate():
    """Valid univariate direct parameters."""
    return {"xi": 0.0, "omega": 1.0, "alpha": 2.0}


@pytest.fixture
def valid_dp_multivariate():
    """Valid multivariate direct parameters."""
    return {
        "xi": np.array([0.0, 1.0]),
        "omega": np.array([[2.0, 0.5], [0.5, 1.0]]),
        "alpha": np.array([2.0, -1.0]),
    }


@pytest.fixture
def valid_cp_univariate():
    """Valid univariate centered parameters."""
    return {
        "mean": 0.0,
        "scale": 1.0,
        "skewness": 0.5,  # Well within bounds
    }


@pytest.fixture
def valid_cp_multivariate():
    """Valid multivariate centered parameters."""
    return {
        "mean": np.array([0.0, 1.0]),
        "scale": np.array([[2.0, 0.5], [0.5, 1.0]]),
        "skewness": np.array([0.5, -0.5]),
    }


class TestDirectParameters:
    """Test direct parameterization validation and handling."""

    def test_valid_univariate(self, valid_dp_univariate):
        """Test creation with valid univariate parameters."""
        dp = DirectParameters(**valid_dp_univariate)
        assert dp.dimension == 1
        assert dp.is_univariate()
        assert dp.tau == 0  # Default value
        assert dp.nu is None  # Default value

    def test_valid_multivariate(self, valid_dp_multivariate):
        """Test creation with valid multivariate parameters."""
        dp = DirectParameters(**valid_dp_multivariate)
        assert dp.dimension == 2
        assert not dp.is_univariate()
        assert_array_equal(dp.xi, valid_dp_multivariate["xi"])
        assert_array_equal(dp.omega, valid_dp_multivariate["omega"])

    def test_invalid_scale(self):
        """Test validation of scale parameter."""
        # Negative scalar omega
        with pytest.raises(ValueError, match="must be positive"):
            DirectParameters(xi=0.0, omega=-1.0, alpha=1.0)

        # Non-symmetric matrix
        with pytest.raises(ValueError, match="must be symmetric"):
            DirectParameters(
                xi=np.array([0.0, 0.0]),
                omega=np.array([[1.0, 0.5], [0.3, 1.0]]),
                alpha=np.array([1.0, 1.0]),
            )

        # Non-positive definite matrix
        with pytest.raises(ValueError, match="must be positive definite"):
            DirectParameters(
                xi=np.array([0.0, 0.0]),
                omega=np.array([[1.0, 2.0], [2.0, 1.0]]),
                alpha=np.array([1.0, 1.0]),
            )

    def test_dimension_mismatch(self):
        """Test validation of parameter dimensions."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            DirectParameters(
                xi=np.array([0.0, 0.0]),
                omega=np.array([[1.0, 0.0], [0.0, 1.0]]),
                alpha=np.array([1.0]),  # Wrong dimension
            )

    def test_optional_parameters(self):
        """Test optional parameters for different distributions."""
        # ST distribution parameters
        dp = DirectParameters(xi=0.0, omega=1.0, alpha=1.0, nu=5.0)
        assert dp.nu == 5.0

        # ESN distribution parameters
        dp = DirectParameters(xi=0.0, omega=1.0, alpha=1.0, tau=2.0)
        assert dp.tau == 2.0

        # Invalid nu
        with pytest.raises(ValueError, match="must be positive"):
            DirectParameters(xi=0.0, omega=1.0, alpha=1.0, nu=-1.0)


class TestCenteredParameters:
    """Test centered parameterization validation and handling."""

    def test_valid_univariate(self, valid_cp_univariate):
        """Test creation with valid univariate parameters."""
        cp = CenteredParameters(**valid_cp_univariate)
        assert cp.dimension == 1
        assert cp.is_univariate()
        assert cp.kurtosis is None  # Default value

    def test_valid_multivariate(self, valid_cp_multivariate):
        """Test creation with valid multivariate parameters."""
        cp = CenteredParameters(**valid_cp_multivariate)
        assert cp.dimension == 2
        assert not cp.is_univariate()
        assert_array_equal(cp.mean, valid_cp_multivariate["mean"])
        assert_array_equal(cp.scale, valid_cp_multivariate["scale"])

    def test_skewness_bounds(self):
        """Test validation of skewness bounds for univariate case."""
        # Valid skewness
        cp = CenteredParameters(mean=0.0, scale=1.0, skewness=0.9)

        # Invalid skewness (too high)
        with pytest.raises(ValueError, match="must be in"):
            CenteredParameters(mean=0.0, scale=1.0, skewness=SN_SKEWNESS_BOUND + 0.1)

        # Invalid skewness (too low)
        with pytest.raises(ValueError, match="must be in"):
            CenteredParameters(mean=0.0, scale=1.0, skewness=-SN_SKEWNESS_BOUND - 0.1)

    def test_type_conversion(self):
        """Test automatic conversion to numpy arrays."""
        # List inputs
        cp = CenteredParameters(
            mean=[0.0, 1.0],
            scale=np.array([[1.0, 0.0], [0.0, 1.0]]),
            skewness=[0.5, -0.5],
        )
        assert isinstance(cp.mean, np.ndarray)
        assert isinstance(cp.skewness, np.ndarray)

        # Integer inputs
        cp = CenteredParameters(mean=0, scale=1, skewness=0)
        assert isinstance(cp.mean, np.ndarray)
        assert isinstance(cp.scale, np.ndarray)
        assert isinstance(cp.skewness, np.ndarray)

    def test_parameters_property(self, valid_cp_univariate):
        """Test parameters dictionary property."""
        cp = CenteredParameters(**valid_cp_univariate)
        params = cp.parameters
        assert isinstance(params, dict)
        assert set(params.keys()) == {"mean", "scale", "skewness"}
        assert params["mean"] == valid_cp_univariate["mean"]
        assert params["skewness"] == valid_cp_univariate["skewness"]
