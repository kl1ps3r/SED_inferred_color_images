from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageSample:
    """Represents one labelable sample with four loaded bands."""

    sample_id: str
    source_path: Path
    bands: dict[str, np.ndarray]
