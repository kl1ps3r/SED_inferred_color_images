from __future__ import annotations

import numpy as np


class BandScaler:
    """Applies percentile clipping and optional asinh stretch."""

    def __init__(self, low_pct: float = 1.0, high_pct: float = 99.0, asinh_a: float = 1.0) -> None:
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.asinh_a = asinh_a

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=float)

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=float)

        low = np.percentile(finite, self.low_pct)
        high = np.percentile(finite, self.high_pct)

        if high <= low:
            high = low + 1e-6

        clipped = np.clip(image, low, high)
        normalized = (clipped - low) / (high - low)

        a = max(self.asinh_a, 1e-6)
        stretched = np.arcsinh(normalized / a) / np.arcsinh(1.0 / a)
        return np.clip(stretched, 0.0, 1.0)
