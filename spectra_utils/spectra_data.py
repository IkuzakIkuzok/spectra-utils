"""Spectra data module.
"""

from typing import Callable, List, Literal, Self, overload
from math import ceil, floor, pi, sin
from numpy import linspace, log10

type Wavelengths = List[float]
type Intensities = List[float]
type Gradient = (
    Literal['linear', 'step', 'sin']
    | Callable[[float, float, float], float]
    | None
)
type AbsorbanceUnit = (
    Literal['Abs', '%T', '%R']
)


class SpectraDataBase():
    """Base class for spectra data.
    """
    @overload
    def __init__(self, filename: str, encoding: str = 'utf-8') -> None:
        """Initializes a new instance of the SpectraDataBase class.

        Args:
            filename (str): Filename to read.
            encoding (str, optional): Encoding of the file.
                Defaults to 'utf-8'.
        """

    @overload
    def __init__(
        self, wavelength: Wavelengths, intensity: Intensities
    ) -> None:
        """Initializes a new instance of the SpectraDataBase class.

        Args:
            wavelength (Wavelength): Wavelength data.
            intensity (Intensities): Intensity data.
        """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a new instance of the SpectraDataBase class.
        """
        self.filename = ''
        self.comment = ''
        self._data: List[str] = []
        self._wavelength: Wavelengths = []
        self._intensity: Intensities = []
        if isinstance(args[0], str):
            filename = args[0]
            encoding = kwargs.get('encoding', 'utf-8')
            self._init_from_file(filename, encoding)
        elif isinstance(args[0], list) and isinstance(args[1], list):
            wavelength = args[0]
            intensity = args[1]
            self._init_from_data(wavelength, intensity)

    def _init_from_file(self, filename: str, encoding: str = 'utf-8') -> None:
        """Initializes a new instance of the SpectraDataBase class.

        Args:
            filename (str): Filename to read.
            encoding (str, optional): Encoding of the file.
                Defaults to 'utf-8'.
        """
        self.filename = filename
        self.encoding = encoding
        self._comment = ''
        self._read_data()
        self._parse_data()

    def _init_from_data(
        self, wavelength: Wavelengths, intensity: Intensities
    ) -> None:
        """Initializes a new instance of the SpectraDataBase class.

        Args:
            wavelength (Wavelength): Wavelength data.
            intensity (Intensities): Intensity data.
        """
        assert len(wavelength) == len(intensity)
        self._wavelength = wavelength
        self._intensity = intensity

    def _read_data(self) -> None:
        """Read data from file.
        """
        assert self.filename
        with open(self.filename, 'r', encoding=self.encoding) as f:
            self._data = f.readlines()

    def _parse_data(self) -> None:
        """Parse data from file.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f'Spectra data ({self.wavelength_min} - {self.wavelength_max})'

    def __repr__(self) -> str:
        return '\n'.join(
            [
                f'Filename: {self.filename}',
                f'Comment: {self.comment}',
                f'Wavelength: {self.wavelength_min} - {self.wavelength_max}',
                f'Intensity: Max. {self.max}',
            ]
        )

    def __len__(self) -> int:
        return len(self._wavelength)

    def __iter__(self):
        return iter(zip(self._wavelength, self._intensity))

    @overload
    def __getitem__(self, key: float) -> float:
        return .0

    @overload
    def __getitem__(self, key: slice) -> Self:
        return self

    def __getitem__(self, key):
        if isinstance(key, float):
            # find the closest wavelength
            if key < self._wavelength[0]:
                return self._intensity[0]
            if key > self._wavelength[-1]:
                return self._intensity[-1]
            for i, w in enumerate(self._wavelength):
                if w > key:
                    return self._intensity[i - 1]
            return self._intensity[self._wavelength.index(key)]

        if isinstance(key, slice):
            return self.__class__(
                self._wavelength[key], self._intensity[key]
            )
        raise TypeError

    @property
    def wavelength(self) -> Wavelengths:
        """Gets wavelength data.
        """
        return self._wavelength

    @property
    def intensity(self) -> Intensities:
        """Gets intensity data.
        """
        return self._intensity

    @property
    def max(self) -> float:
        """Gets max intensity.
        """
        return max(self._intensity)

    @property
    def wavelength_min(self) -> float:
        """Gets min wavelength.
        """
        return min(self._wavelength)

    @property
    def wavelength_max(self) -> float:
        """Gets max wavelength.
        """
        return max(self._wavelength)

    def normalize(self, denominator: float | None = None) -> Self:
        """Normalizes intensity data.

        Args:
            denominator (float, optional):
                Denominator to normalize.
                If `None`, use max intensity as denominator. Defaults to None.

        Returns:
            Self: Normalized spectra data.
        """
        if denominator is None:
            denominator = max(self._intensity)
        self._intensity = [i / denominator for i in self._intensity]

        return self

    def concat(
        self, other: Self,
        step: float | None = None, gradient: Gradient = 'sin'
    ) -> Self:
        """Concatenates two spectra data.

        Args:
            other (Self): Other spectra data.
            step (float, optional): The step size of the concatenated data.
            gradient (Gradient, optional):
                The gradient mode to use when concatenating.

        Returns:
            Self: Concatenated spectra data.
        """

        left = self if self.wavelength_min < other.wavelength_min else other
        right = self if self.wavelength_min >= other.wavelength_min else other

        if step is None:
            step = left.wavelength[1] - left.wavelength[0]
        offset = left.wavelength[-1] % step
        left_l = ceil(left.wavelength[0] / step) * step + offset
        left_u = floor(left.wavelength[-1] / step) * step + offset
        right_l = ceil(right.wavelength[0] / step) * step + offset
        right_u = floor(right.wavelength[-1] / step) * step + offset

        wl_left_new = list(
            linspace(left_l, left_u, int((left_u - left_l) / step) + 1)
        )
        wl_right_new = list(
            linspace(right_l, right_u, int((right_u - right_l) / step) + 1)
        )
        i_left_new = left.resample(wl_left_new).intensity
        i_right_new = right.resample(wl_right_new).intensity

        overlap_l, overlap_u = wl_right_new[0], wl_left_new[-1]
        overlap = overlap_u - overlap_l
        n_overlap = int(overlap / step)

        length = int((right_u - left_l) / step) + 1
        wavelengths = list(linspace(left_l, right_u, length))
        intensities = [.0] * length

        def get_gradient_func() -> Callable[[float, float, float], float]:
            if callable(gradient):

                def wrapper(x: float, i1: float, i2: float) -> float:
                    # this assertion is redundant, but it makes pyright happy
                    assert callable(gradient)

                    r = (x - overlap_l) / overlap
                    return gradient(r, i1, i2)

                return wrapper

            if gradient == 'linear':
                def linear_gradient(x: float, i1: float, i2: float) -> float:
                    r = (x - overlap_l) / overlap
                    return i1 + (i2 - i1) * r

                return linear_gradient

            if gradient == 'step':
                mid = (overlap_u + overlap_l) / 2

                def step_gradient(x: float, i1: float, i2: float) -> float:
                    if x < mid:
                        return i1
                    if x > mid:
                        return i2
                    return (i1 + i2) / 2

                return step_gradient

            if gradient == 'sin':

                def sin_gradient(x: float, i1: float, i2: float) -> float:
                    r = ((x - overlap_l) / overlap - 0.5) * pi
                    return i1 + (i2 - i1) * ((sin(r) + 1) / 2)

                return sin_gradient

            def no_gradient(_: float, i1: float, i2: float) -> float:
                return (i1 + i2) / 2

            return no_gradient

        grad_func = get_gradient_func()
        ratios = [
            left.intensity[-i] / i_right_new[i] for i in range(n_overlap)
        ]
        ratio = sum(ratios) / len(ratios)
        i_right_new = [i * ratio for i in i_right_new]

        right_offset = len(wl_left_new) - n_overlap - 1
        for i, w in enumerate(wavelengths):
            if w < overlap_l:
                intensities[i] = i_left_new[i]
            elif w > overlap_u:
                intensities[i] = i_right_new[i - right_offset]
            else:
                intensities[i] = grad_func(
                    w, i_left_new[i], i_right_new[i - right_offset]
                )

        return self.__class__(wavelengths, intensities)

    def resample(self, wavelength: Wavelengths) -> Self:
        """Resamples data to new wavelength.

        Args:
            wavelength (Wavelength): New wavelength.

        Returns:
            Self: Resampled spectra data.
        """

        def search_wavelength(wavelength: float) -> int:
            """Search for a wavelength in the spectrum.

            Args:
                wavelength (float): The wavelength to search for.

            Returns:
                i (int): The index of the wavelength in the spectrum.
            """
            for i, w in enumerate(self._wavelength):
                if w > wavelength:
                    return i
            return -1

        new_intensities = [.0] * len(wavelength)
        for idx, wl in enumerate(wavelength):
            i = search_wavelength(wl)
            if i == 0:
                new_intensities[idx] = self._intensity[0]
            elif i == len(self._wavelength):
                new_intensities[idx] = self._intensity[-1]
            else:
                new_intensities[idx] = (
                    self._intensity[i - 1]
                    + (self._intensity[i] - self._intensity[i - 1]) *
                    (wl - self._wavelength[i - 1]) /
                    (self._wavelength[i] - self._wavelength[i - 1])
                )

        new_data = self.__class__(wavelength, new_intensities)
        new_data.filename = self.filename
        new_data.comment = self.comment
        return new_data


class NanoLog(SpectraDataBase):
    """NanoLog data class.
    """
    @overload
    def __init__(self, filename: str) -> None:
        """Initializes a new instance of the NanoLog class from file.

        Args:
            filename (str): The path to the file.
        """
        super().__init__(filename, encoding='utf-8')
        self._parse_data()

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
        super().__init__(wavelength, intensity)

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


class UH4150(SpectraDataBase):
    """UH4150 data class.
    """
    @overload
    def __init__(self, filename: str) -> None:
        """Initializes a new instance of the UH4150 class from file.

        Args:
            filename (str): The path to the file.
        """
        super().__init__(filename, encoding='utf-8')
        self._parse_data()

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
        super().__init__(wavelength, intensity)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unit: AbsorbanceUnit = 'Abs'  # pyright: ignore

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
            self._wavelength = [float(d[0].strip()) for d in data]
            self._intensity = [float(d[1].strip()) for d in data]
            break

    def __str__(self) -> str:
        return \
            f'Absorbance data ({self.wavelength_min} - {self.wavelength_max})'

    def absorbance(self) -> Intensities:
        """Gets absorbance data.
        """
        if self.unit == 'Abs':
            return self._intensity
        return [-log10(i / 100) for i in self._intensity]

    def transmittance(self) -> Intensities:
        """Gets transmittance data.
        """
        if self.unit == '%T':
            return self._intensity
        return [10 ** (-i) for i in self._intensity]
