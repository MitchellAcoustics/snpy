# snpy

A Python interface for skew-normal distributions powered by R's `sn` package.

> **Note**
> This package is currently under development and not yet released on PyPI. Stay tuned for the initial release!

## Overview

`snpy` provides a Pythonic wrapper around the powerful R package `sn` for working with skew-normal distributions. It aims to make the functionality of `sn` easily accessible to Python users while handling the R dependencies seamlessly.

<!-- ## Installation

```bash
# Basic installation
pip install snpy

# With R integration support
pip install snpy[r]
``` -->

## Features

- Python interface for skew-normal distribution functions
- Seamless integration with R's `sn` package
- NumPy and Pandas compatibility
- Easy-to-use API for:
  
  - Probability density functions
  - Cumulative distribution functions
  - Random number generation
  - Parameter estimation

## Quick Start

```python
import snpy

# Create skew-normal distribution
x = snpy.rvs(location=0, scale=1, shape=5, size=1000)

# Calculate probability density
pdf = snpy.pdf(x)

# Fit skew-normal to data
params = snpy.fit(x)
```

## Dependencies

- Python ≥ 3.10
- pandas ≥ 2.2.3
- rpy2 ≥ 3.5.16 (optional)
- R with `sn` package (optional)

<!-- ## Documentation

For detailed documentation, visit: https://snpy.readthedocs.io/ -->

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

MIT License

## Citation

If you use this software, please cite both this package and the underlying R `sn` package.

## Acknowledgments

This package was inspired by [pymer4](https://github.com/ejolly/pymer4) and builds upon the R `sn` package by Adelchi Azzalini.