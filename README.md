# <img src="assets/logo.svg" alt="logo" width="128" height="128" align="middle"> pyUTSAlgorithms (Unevenly Spaced Time Series Algorithms)

![PyPI - Version](https://img.shields.io/pypi/v/pyUTSAlgorithms)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyUTSAlgorithms)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/pyUTSAlgorithms)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/pyUTSAlgorithms/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/pyUTSAlgorithms)

Unevenly Spaced Time Series (UTS) algorithms for moving averages, rolling operators, and peak detection.
This repository provides a comprehensive set of tools to handle time series where observations are not recorded at regular intervals.

This library started as a conversion of the C++ code from [andreas50/utsAlgorithms](https://github.com/andreas50/utsAlgorithms). It has since been expanded with additional algorithms specifically designed for unevenly spaced data.

## Features

- **Exponential Moving Averages (EMA):** Supports `last`, `next`, and `linear` interpolation schemes for numerical stability in UTS.
- **Simple Moving Averages (SMA):** Efficient rolling window averages for uneven grids using `last`, `next`, and `linear` interpolation.
- **Gradient Computation:** Central first and second-order derivatives, and arbitrary order derivatives via Fornberg's algorithm.
- **Peak and Valley Detection:**
    - Basic peak/valley identification.
    - Significant peaks based on Z-score, Kneedle algorithm, or peak prominence.
- **Rolling Operators:** Generic rolling statistics (mean, variance, max, min, product, etc.) on unevenly spaced time intervals.
- **Thresholding & Z-Score:** Specialized utilities (like ISODATA) for data analysis and noise reduction in UTS.

## Quick Start

```python
import numpy as np
import uts.ema as ema

# Sample unevenly spaced data: (time, value)
data = np.array([
    [0.0, 10.0],
    [1.5, 12.0],
    [2.1, 11.0],
    [4.5, 15.0],
    [5.0, 14.0]
])

# Compute EMA with a tau (half-life) of 1.0 using linear interpolation
results = ema.linear(data, tau=1.0)

print("Original Data:\n", data)
print("EMA Results:\n", results)
```

## Running Unit Tests

Several unit tests were written to validate corner cases and ensure accuracy.
The unit tests use the standard [unittest](https://docs.python.org/3/library/unittest.html) framework.

```bash
python -m unittest
```

## Documentation

The library is documented using Google-style docstrings. Detailed API documentation is available [here](https://mariolpantunes.github.io/pyUTSAlgorithms/).

To generate the documentation locally:

```bash
pip install pdoc
pdoc --math -d google -o docs src/uts
```

## Installation

### From PyPI
Install the latest version directly from PyPI:

```bash
pip install pyUTSAlgorithms
```

### From Source
To install the library locally:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install .
```

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
