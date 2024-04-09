"""Setup script for spectra_utils package.
"""

from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='spectra_utils',
    version='1.1.0',
    description='Spectra utils to process experimental data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Kazuki Kohzuki',
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    install_requires=[
        'numpy',
    ],
)
