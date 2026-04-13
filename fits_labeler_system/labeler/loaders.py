from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from astropy.io import fits

from .models import ImageSample


class BaseFitsLoader(ABC):
    """Abstract interface so loading strategies can be swapped later."""

    @abstractmethod
    def discover(self) -> list[str]:
        """Return sample identifiers available from this loader."""

    @abstractmethod
    def load(self, sample_id: str) -> ImageSample:
        """Load one sample and return all four bands."""


def _read_fits_2d(path: Path, hdu: int | str = 0) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        hdu_obj = hdul[hdu]
        data = getattr(hdu_obj, "data", None)

    if data is None:
        raise ValueError(f"No data found in HDU {hdu} for {path}")

    arr = np.asarray(data, dtype=float)

    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D data in {path}, got shape {arr.shape}")

    return arr


class SeparateFilesByStemLoader(BaseFitsLoader):
    """Loads one sample from four files that share a common stem."""

    def __init__(self, root_dir: Path, band_suffixes: dict[str, str]) -> None:
        if len(band_suffixes) != 4:
            raise ValueError("Exactly four bands are required")

        self.root_dir = root_dir
        self.band_suffixes = band_suffixes
        self._bands = list(band_suffixes.keys())

    def discover(self) -> list[str]:
        reference_band = self._bands[0]
        reference_suffix = self.band_suffixes[reference_band]

        sample_ids: list[str] = []
        for path in sorted(self.root_dir.glob(f"*{reference_suffix}")):
            stem = path.name[: -len(reference_suffix)]
            if self._all_band_files_exist(stem):
                sample_ids.append(stem)
        return sample_ids

    def load(self, sample_id: str) -> ImageSample:
        band_data: dict[str, np.ndarray] = {}

        for band, suffix in self.band_suffixes.items():
            file_path = self.root_dir / f"{sample_id}{suffix}"
            band_data[band] = _read_fits_2d(file_path)

        source = self.root_dir / f"{sample_id}{self.band_suffixes[self._bands[0]]}"
        return ImageSample(sample_id=sample_id, source_path=source, bands=band_data)

    def _all_band_files_exist(self, stem: str) -> bool:
        for suffix in self.band_suffixes.values():
            if not (self.root_dir / f"{stem}{suffix}").exists():
                return False
        return True


class MultiHduFitsLoader(BaseFitsLoader):
    """Loads four bands from different HDUs inside a single FITS file."""

    def __init__(self, root_dir: Path, pattern: str, hdu_map: dict[str, int | str]) -> None:
        if len(hdu_map) != 4:
            raise ValueError("Exactly four bands are required")

        self.root_dir = root_dir
        self.pattern = pattern
        self.hdu_map = hdu_map

    def discover(self) -> list[str]:
        return [p.name for p in sorted(self.root_dir.glob(self.pattern))]

    def load(self, sample_id: str) -> ImageSample:
        file_path = self.root_dir / sample_id
        band_data = {band: _read_fits_2d(file_path, hdu=hdu) for band, hdu in self.hdu_map.items()}
        return ImageSample(sample_id=sample_id, source_path=file_path, bands=band_data)
