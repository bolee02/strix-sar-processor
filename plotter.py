"""Range spectrum and power overview for a StriX SICD SLC."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.sicd_elements.SICD import SICDType

from windowing import (
    apply_range_hanning,
    multilooked_power,
    occupied_band,
    range_hanning_window,
)
from loader import describe, open_slc

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def mean_range_spectrum(
    data: np.ndarray,
    sample_spacing: float,
    sgn: int,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean range power spectrum of a `(range, azimuth)` block."""

    spectrum = np.fft.fft(data, axis=0)
    mean_power = np.mean(np.abs(spectrum) ** 2, axis=1, dtype=np.float64)

    # With Sgn = -1 this is fftfreq. See _band_bins.
    frequency = -sgn * np.fft.fftfreq(data.shape[0], d=sample_spacing)
    order = np.argsort(frequency)
    frequency = frequency[order]
    mean_power = mean_power[order]
    low, high = band
    margin = 0.05 * (high - low)
    inside = (frequency >= low + margin) & (frequency <= high - margin)
    reference = np.median(mean_power[inside])
    decibels = 10.0 * np.log10(mean_power / reference)
    return frequency, decibels


def center_column_block(
    reader: SICDTypeReader,
    sicd: SICDType,
    n_columns: int = 1024,
) -> tuple[np.ndarray, int]:
    """Read `n_columns` around the SCP column."""

    _, n_cols = reader.get_data_size_as_tuple()[0]
    center = int(sicd.ImageData.SCPPixel.Col)
    start = max(0, min(center - n_columns // 2, n_cols - n_columns))
    stop = min(n_cols, start + n_columns)
    return np.asarray(reader[:, start:stop]), start


def plot_mean_range_spectrum(
    frequency: np.ndarray,
    decibels: np.ndarray,
    sicd: SICDType,
    output_path: Path,
    n_columns: int,
) -> None:
    """Save the mean range spectrum with support and bandwidth marked."""

    row = sicd.Grid.Row
    nyquist = 0.5 / row.SS
    center = 0.0 if row.DeltaKCOAPoly is None else float(np.asarray(row.DeltaKCOAPoly.Coefs)[0, 0])
    half_bw = row.ImpRespBW / 2

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    ax.plot(frequency, decibels, color="0.1", lw=1.0)
    ax.axvspan(row.DeltaK1, row.DeltaK2, color="C0", alpha=0.18, label="DeltaK1–DeltaK2")
    ax.axvspan(
        center - half_bw,
        center + half_bw,
        facecolor="none",
        edgecolor="C1",
        hatch="///",
        label="DeltaKCOA ± ImpRespBW/2",
    )
    ax.axvline(center, color="C3", ls=":", lw=1.2, label="DeltaKCOA at SCP")
    ax.axvline(-nyquist, color="0.45", ls="--", lw=0.8)
    ax.axvline(nyquist, color="0.45", ls="--", lw=0.8, label="Nyquist (±1 / 2 SS)")

    ax.set_xlim(-nyquist * 1.02, nyquist * 1.02)
    floor = max(float(np.min(decibels)), -70.0)
    ax.set_ylim(floor, max(3.0, float(np.max(decibels)) + 1.0))
    ax.set_xlabel("Range spatial frequency offset from KCtr (cycles/m)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Mean range spectrum, {n_columns} columns")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", frameon=True, facecolor="white", edgecolor="0.8", framealpha=1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_db_overview(
    power: np.ndarray,
    output_path: Path,
    row_step: int,
    col_step: int,
) -> None:
    """Save multilooked power in dB, with row 0 at the top."""

    peak = float(np.max(power))
    decibels = 10.0 * np.log10(np.maximum(power, peak * 1e-6))
    low, high = np.percentile(decibels, [2, 98])

    n_rows = power.shape[0] * row_step
    n_cols = power.shape[1] * col_step

    fig, ax = plt.subplots(figsize=(11, 4.4))

    image = ax.imshow(
        decibels,
        origin="upper",
        extent=(0, n_cols, n_rows, 0),
        vmin=low,
        vmax=high,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Power (dB), near range at top")
    fig.colorbar(image, ax=ax, label="Power (dB)", fraction=0.025, pad=0.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_hanning_window(
    frequency: np.ndarray,
    decibels: np.ndarray,
    window: np.ndarray,
    output_path: Path,
) -> None:
    """Save the mean range spectrum before and after the Hanning window."""

    weighted = decibels + 10.0 * np.log10(window ** 2 + 1e-12)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(frequency, decibels, color="0.1", lw=1.0, label="Unweighted")
    ax.plot(frequency, weighted, color="C1", lw=1.2, label="Hanning")
    floor = max(float(np.min(decibels)), -70.0)
    top = max(float(np.max(decibels)), float(np.max(weighted[window > 1e-3]))) + 1.0
    ax.set_ylim(floor, top)
    ax.set_xlabel("Range spatial frequency offset from KCtr (cycles/m)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Range spectrum after Hanning")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower center",
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def render(
    reader: SICDTypeReader,
    sicd: SICDType,
    output_dir: Path = OUTPUT_DIR,
    n_columns: int = 1024,
) -> None:
    """Write the spectrum, Hanning comparison and power overview."""

    block, start = center_column_block(reader, sicd, n_columns)
    row = sicd.Grid.Row
    band = occupied_band(row)
    frequency, decibels = mean_range_spectrum(block, row.SS, row.Sgn, band)
    spectrum_path = output_dir / "mean_range_spectrum.png"
    plot_mean_range_spectrum(frequency, decibels, sicd, spectrum_path, n_columns=block.shape[1])
    print(f"Range spectrum from columns {start}:{start + block.shape[1]}, saved to {spectrum_path}")

    fft_frequency = -row.Sgn * np.fft.fftfreq(block.shape[0], d=row.SS)
    order = np.argsort(fft_frequency)
    window = range_hanning_window(block.shape[0], row.SS, row.Sgn, *band)
    window_path = output_dir / "hanning_window.png"
    plot_hanning_window(frequency, decibels, window[order], window_path)
    print(f"Hanning window saved to {window_path}")

    row_step, col_step = 4, 16
    power = multilooked_power(reader, sicd, row_step=row_step, col_step=col_step)

    overview_path = output_dir / "db_overview.png"
    plot_db_overview(power, overview_path, row_step=row_step, col_step=col_step)
    print(f"Power overview saved to {overview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the range spectrum and dB overview of an SLC.")
    parser.add_argument("slc", type=Path, help="path to the SLC .nitf file")
    args = parser.parse_args()

    reader, sicd = open_slc(args.slc)
    try:
        describe(reader, sicd)
        render(reader, sicd)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
