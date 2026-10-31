"""Range-spectrum windowing and multilook for a StriX SLC."""

import numpy as np
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType


def multilook(power: np.ndarray, row_step: int, col_step: int) -> np.ndarray:
    """Average power over non-overlapping blocks of `row_step` by `col_step`."""

    # Trim edges so the array divides evenly.
    r = power.shape[0] // row_step * row_step
    c = power.shape[1] // col_step * col_step

    p = power[:r, :c].reshape(r // row_step, row_step, c // col_step, col_step)
    return p.mean(axis=(1, 3))


def multilooked_power(
    reader: SICDTypeReader,
    sicd: SICDType,
    row_step: int = 4,
    col_step: int = 16,
    block_cols: int = 2048,
) -> np.ndarray:
    """Multilooked power, averaged over each `row_step` by `col_step` block."""

    n_rows = sicd.ImageData.NumRows
    n_cols = sicd.ImageData.NumCols

    # Block width is a multiple of col_step so looks never straddle two reads.
    block_cols = col_step * max(1, block_cols // col_step)

    power = np.empty((n_rows // row_step, n_cols // col_step), dtype=np.float32)
    out_col = 0
    for start in range(0, n_cols, block_cols):
        stop = min(n_cols, start + block_cols)
        block = np.asarray(reader[:, start:stop])
        looked = multilook(np.abs(block) ** 2, row_step, col_step)
        power[:, out_col : out_col + looked.shape[1]] = looked
        out_col += looked.shape[1]

    return power


def occupied_band(direction) -> tuple[float, float]:
    """Return `DeltaKCOA ± ImpRespBW/2` for one grid direction."""

    coefs = direction.DeltaKCOAPoly
    center = 0.0 if coefs is None else float(np.asarray(coefs.Coefs)[0, 0])
    half = 0.5 * float(direction.ImpRespBW)
    return center - half, center + half


def _band_bins(n: int, sample_spacing: float, sgn: int, low: float, high: float) -> np.ndarray:
    # With Sgn = -1, numpy's fftfreq is already the SICD frequency.
    frequency = -sgn * np.fft.fftfreq(n, d=sample_spacing)
    inside = np.flatnonzero((frequency >= low) & (frequency <= high))
    return inside[np.argsort(frequency[inside])]


def rectangular_window(
    n: int,
    sample_spacing: float,
    sgn: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Ones on `[low, high]` and zero outside, in `np.fft.fft` order."""

    window = np.zeros(n, dtype=np.float64)
    window[_band_bins(n, sample_spacing, sgn, low, high)] = 1.0
    return window


def range_hanning_window(
    n: int,
    sample_spacing: float,
    sgn: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Hanning weights on `[low, high]`, in `np.fft.fft` order."""

    inside = _band_bins(n, sample_spacing, sgn, low, high)
    window = np.zeros(n, dtype=np.float64)
    if inside.size == 0:
        return window
    window[inside] = np.hanning(inside.size)

    # Scale the window so the mean of `w²` on the occupied band is 1.
    window *= np.sqrt(inside.size / np.sum(window ** 2))
    return window


def apply_range_window(data: np.ndarray, window: np.ndarray) -> np.ndarray:
    """FFT each range line, multiply by `window`, and transform back."""

    spectrum = np.fft.fft(data, axis=0)
    weighted = np.fft.ifft(spectrum * window[:, np.newaxis], axis=0)
    return weighted.astype(data.dtype)


def apply_range_hanning(
    data: np.ndarray,
    sample_spacing: float,
    sgn: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Weight each range line: FFT, Hanning on the central band, then IFFT."""

    window = range_hanning_window(data.shape[0], sample_spacing, sgn, low, high)
    return apply_range_window(data, window)
