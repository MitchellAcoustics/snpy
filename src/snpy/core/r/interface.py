"""R interface singleton for managing R integration.

This module provides a singleton class that manages the R interface,
including package loading, data conversion, and error handling. It uses
rpy2 to interact with R and provides convenient methods for common operations.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.rinterface_lib.callbacks import logger
from rpy2.rinterface_lib.embedded import RRuntimeError

# Configure rpy2 logging
logger.setLevel(logging.WARNING)


class RPackageError(Exception):
    """Error raised when there are issues with R packages."""

    pass


class RInterface:
    """Singleton class managing R interface and package loading.

    This class follows the singleton pattern to ensure only one R interface
    exists. It handles:
    - R package loading and version checking
    - Data conversion between Python and R
    - Error handling for R operations
    - Common R utility functions

    Attributes
    ----------
    sn : rpy2.robjects.packages.Package
        The R 'sn' package
    stats : rpy2.robjects.packages.Package
        R 'stats' package for additional statistical functions
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize R interface if not already initialized."""
        if not self._initialized:
            self._initialize()

    def _initialize(self):
        """Initialize R interface and load required packages."""
        # Activate automatic conversion
        numpy2ri.activate()
        pandas2ri.activate()

        try:
            # Import required R packages
            self.sn = importr("sn")
            self.stats = importr("stats")

            # Verify package versions
            self._check_package_versions()

            # Store commonly used R functions
            self._cache_r_functions()

            self._initialized = True

        except PackageNotInstalledError as e:
            raise RPackageError(
                f"Required R package not installed: {e.package}. "
                "Please install required R packages. See installation guide."
            ) from e
        except Exception as e:
            raise RPackageError(
                "Failed to initialize R interface. Ensure R is installed "
                "and accessible."
            ) from e

    def _check_package_versions(self):
        """Verify R package versions meet requirements."""
        try:
            sn_version = str(self.sn._get_namespace_env()["packageVersion"])
            # TODO: Add version comparison logic
        except Exception as e:
            logging.warning(f"Could not verify R package versions: {e}")

    def _cache_r_functions(self):
        """Cache commonly used R functions for efficiency."""
        self._r_dsn = self.sn.dsn
        self._r_dmsn = self.sn.dmsn
        self._r_cp2dp = self.sn.cp2dp
        self._r_dp2cp = self.sn.dp2cp

    def ensure_initialized(self):
        """Ensure R interface is initialized."""
        if not self._initialized:
            raise RuntimeError("R interface not initialized")

    def convert_to_r_vector(self, x: Union[float, np.ndarray]) -> ro.Vector:
        """Convert Python numeric types to R vector."""
        if isinstance(x, (int, float)):
            return ro.FloatVector([x])
        elif isinstance(x, np.ndarray):
            if x.ndim > 1:
                raise ValueError("Array must be 1-dimensional")
            return ro.FloatVector(x.ravel())
        else:
            raise TypeError(f"Cannot convert type {type(x)} to R vector")

    def convert_to_r_matrix(self, x: np.ndarray) -> ro.Matrix:
        """Convert numpy array to R matrix."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim != 2:
            raise ValueError("Array must be 2-dimensional")

        return ro.r.matrix(ro.FloatVector(x.ravel()), nrow=x.shape[0], ncol=x.shape[1])

    def safe_call(self, func: Any, *args, **kwargs) -> Any:
        """Safely call R function with error handling."""
        try:
            self.ensure_initialized()
            return func(*args, **kwargs)
        except RRuntimeError as e:
            error_msg = str(e)
            if "non-conformable arrays" in error_msg:
                raise ValueError("Non-conformable arrays in R operation")
            elif "singular matrix" in error_msg:
                raise ValueError("Singular matrix in R operation")
            else:
                raise RPackageError(f"R error: {error_msg}") from e

    def convert_params_to_r(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter dictionary to R-compatible format."""
        r_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2:
                    r_params[key] = self.convert_to_r_matrix(value)
                else:
                    r_params[key] = self.convert_to_r_vector(value)
            else:
                r_params[key] = self.convert_to_r_vector(value)
        return r_params

    # Utility methods for common operations
    def cp2dp(self, cp_params: Dict[str, Any], family: str = "SN") -> Dict[str, Any]:
        """Convert CP parameters to DP using R's cp2dp."""
        r_params = self.convert_params_to_r(cp_params)
        result = self.safe_call(self._r_cp2dp, r_params, family=family)
        # TODO: Convert result back to Python format
        return result

    def dp2cp(self, dp_params: Dict[str, Any], family: str = "SN") -> Dict[str, Any]:
        """Convert DP parameters to CP using R's dp2cp."""
        r_params = self.convert_params_to_r(dp_params)
        result = self.safe_call(self._r_dp2cp, r_params, family=family)
        # TODO: Convert result back to Python format
        return result
