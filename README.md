
# spectra-utils

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/IkuzakIkuzok/spectra-utils/blob/main/LICENSE)

Utilities for working with spectra data.

## Overview

`spectra_utils` is a Python package for working with spectra data.
It provides a class `NanoLog` for reading emission spectra
exported from Horiba NanoLog software.
It also provides a class `UH4150` for reading absorption spectra
exported from Hitachi UH4150 software.
This class is compatible with U4100`, which is the older version of UH4150.

## Installation

```bash
pip install spectra_utils@git+https://github.com/IkuzakIkuzok/spectra-utils.git
```

## Usage

```python
import spectra_utils as su
import matplotlib.pyplot as plt

# Read spectra data
vis = su.NanoLog("data/vis.csv")
nir = su.NanoLog("data/nir.csv")
concat = su.concat(vis, nir)

# Plot spectra
plt.scatter(vis.wavelength, vis.intensity, s=1, c="r")
plt.scatter(nir.wavelength, nir.intensity, s=1, c="b")
plt.plot(concat.wavelength, concat.intensity, c="k")
```

## License

This project is licensed under the MIT License.
