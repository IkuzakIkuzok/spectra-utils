"""Emission data module.
"""

from typing import overload

from .spectra_data import SpectraDataBase, Wavelengths, Intensities


class NanoLog(SpectraDataBase):
    """NanoLog data class.
    """
    @overload
    def __init__(self, filename: str) -> None:
        """Initializes a new instance of the NanoLog class from file.

        Args:
            filename (str): The path to the file.
        """

    @overload
    def __init__(
        self, wavelength: Wavelengths, intensity: Intensities
    ) -> None:
        """Initializes a new instance of the NanoLog class
        from wavelength and intensity data.

        Args:
            wavelength (Wavelengths): The wavelength data.
            intensity (Intensities): The intensity data.
        """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _parse_data(self) -> None:
        """Parse data from file.
        """
        data = list(zip(*[d.split('\t') for d in self._data[2:]]))
        self._wavelength = [float(d) for d in data[0]]
        self._intensity = [float(d) for d in data[1]]

    def __str__(self) -> str:
        return f'Emission data ({self.wavelength_min} - {self.wavelength_max})'
