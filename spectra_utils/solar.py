"""Solar spectrum data.
"""

from pathlib import Path
from typing import List, Literal, Tuple

type SpectrumType = Literal['AM0', 'AM1.5G', 'AM1.5D']


class SolarSpectra():
    """Solar spectrum data.
    """

    __filename = Path(__file__).parent / 'ASTMG173.csv'

    def __init__(self, spectrum_type: SpectrumType = 'AM1.5G'):
        index = ['AM0', 'AM1.5G', 'AM1.5D'].index(spectrum_type)
        self.__wavelength, self.__irradiance = self.__load(index)

    def __load(self, index: int) -> Tuple[List[float], List[float]]:
        with open(self.__filename, 'r', encoding='utf-8') as f:
            data = f.readlines()[2:]

        values = [tuple(map(float, line.split(','))) for line in data]
        wavelength = [value[0] for value in values]
        irradiance = [value[index + 1] for value in values]
        return wavelength, irradiance

    @property
    def wavelength(self) -> List[float]:
        """Wavelength [nm].
        """
        return self.__wavelength

    @property
    def irradiance(self) -> List[float]:
        """Irradiance [W/m^2/nm].
        """
        return self.__irradiance
