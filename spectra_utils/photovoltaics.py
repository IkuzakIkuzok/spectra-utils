"""Photovoltaics module.
"""

from typing import List, overload
import numpy as np

from .spectra_data import SpectraDataBase, Wavelengths, Intensities
from .lazy import InitOnAccess
from .solar import SolarSpectra, SpectrumType


class ECT250D(SpectraDataBase):
    """ECT250D data class.
    """

    _am0: SolarSpectra = InitOnAccess(SolarSpectra, 'AM0')
    _am15g: SolarSpectra = InitOnAccess(SolarSpectra, 'AM1.5G')
    _am15d: SolarSpectra = InitOnAccess(SolarSpectra, 'AM1.5D')

    @overload
    def __init__(self, filename: str) -> None:
        """Initializes a new instance of the ECT250D class from file.

        Args:
            filename (str): The path to the file.
        """

    @overload
    def __init__(
        self, wavelength: Wavelengths, intensity: Intensities
    ) -> None:
        """Initializes a new instance of the ECT250D class
        from wavelength and intensity data.

        Args:
            wavelength (Wavelengths): The wavelength data.
            intensity (Intensities): The intensity data.
        """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, encoding='shift_jis', **kwargs)

    def _parse_data(self) -> None:
        """Parse data from file.
        """
        lines = list(map(lambda x: x.split(','), self._data[24:]))
        self._wavelength = [float(d[0].strip()) for d in lines]
        self._intensity = [float(d[-1].strip()) for d in lines]

    def __str__(self) -> str:
        return \
            f'EQE data ({self.wavelength_min} - {self.wavelength_max})'

    def j_sc(self, am: SpectrumType = 'AM1.5G') -> List[float]:
        """Calculates short-circuit current density from EQE spectrum.

        Args:
            am (SpectrumType, optional):
                The spectrum type to use.
                Defaults to 'AM1.5G'.

        Returns:
            List[float]: Short-circuit current density.
        """
        sun = {
            'AM0': self._am0,
            'AM1.5G': self._am15g,
            'AM1.5D': self._am15d
        }[am]

        step = self.wavelength[1] - self.wavelength[0]
        w = int(step * 2)

        move_mean_sun = np.convolve(
            sun.irradiance, np.ones(w) / w, 'same'
        )
        std_sun = [
            s / 1e4 * 1e3
            for w, s in zip(sun.wavelength, move_mean_sun)
            if w in self.wavelength
        ]

        j = [i * s / 100 for i, s in zip(self._intensity, std_sun)]
        s = [.0] * len(j)
        s[0] = j[0] * self.wavelength[0] / self.WAVELENGTH_TO_ENERGY
        for i in range(1, len(j)):
            x2 = self.wavelength[i]
            x1 = self.wavelength[i - 1]
            y2 = j[i] * x2 / self.WAVELENGTH_TO_ENERGY
            y1 = j[i - 1] * x1 / self.WAVELENGTH_TO_ENERGY
            s[i] = s[i - 1] + (x2 - x1) * (y2 + y1) / 2

        return s
