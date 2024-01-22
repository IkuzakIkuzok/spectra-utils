"""Absorption data class.
"""

from typing import Literal, overload
import numpy as np

from .spectra_data import \
    SpectraDataBase, Wavelengths, Intensities

type AbsorbanceUnit = (
    Literal['Abs', '%T', '%R']
)


class UH4150(SpectraDataBase):
    """UH4150 data class.
    """
    @overload
    def __init__(self, filename: str) -> None:
        """Initializes a new instance of the UH4150 class from file.

        Args:
            filename (str): The path to the file.
        """

    @overload
    def __init__(
        self, wavelength: Wavelengths, intensity: Intensities
    ) -> None:
        """Initializes a new instance of the UH4150 class
        from wavelength and intensity data.

        Args:
            wavelength (Wavelengths): The wavelength data.
            intensity (Intensities): The intensity data.
        """

    def __init__(self, *args, **kwargs) -> None:
        self.unit: AbsorbanceUnit = 'Abs'  # pyright: ignore
        super().__init__(*args, encoding='shift_jis', **kwargs)

    def _parse_data(self) -> None:
        """Parse data from file.
        """
        lines = list(map(lambda x: x.split('\t'), self._data))
        for i, line in enumerate(lines):
            if not line[0] == 'nm':
                continue
            u = line[1].strip()
            if u not in ['Abs', '%T', '%R']:
                raise ValueError('Unknown unit')
            self.unit: AbsorbanceUnit = u  # pyright: ignore
            data = lines[i + 1:-1]
            self._wavelength = [float(d[0].strip()) for d in data[::-1]]
            self._intensity = [float(d[1].strip()) for d in data[::-1]]
            break

    def __str__(self) -> str:
        return \
            f'Absorbance data ({self.wavelength_min} - {self.wavelength_max})'

    def absorbance(self) -> Intensities:
        """Gets absorbance data.
        """
        if self.unit == 'Abs':
            return self._intensity
        return [-np.log10(i / 100) for i in self._intensity]

    def transmittance(self) -> Intensities:
        """Gets transmittance data.
        """
        if self.unit == '%T':
            return self._intensity
        return [10 ** (-i) for i in self._intensity]
