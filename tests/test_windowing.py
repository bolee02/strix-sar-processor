"""Correctness checks for the range Hanning window."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from windowing import (
    apply_range_hanning,
    apply_range_window,
    multilook,
    range_hanning_window,
    rectangular_window,
)

N = 512
SAMPLE_SPACING = 1.5
SGN = -1
BAND = (-0.25, 0.25)


def _sicd_frequency(
    n: int = N,
    sample_spacing: float = SAMPLE_SPACING,
    sgn: int = SGN,
) -> np.ndarray:
    """SICD frequency of each FFT bin.

    With Sgn = -1, numpy's fftfreq is already the SICD frequency, so the bin
    frequency is -Sgn * fftfreq.
    """

    return -sgn * np.fft.fftfreq(n, d=sample_spacing)


def _inside(
    n: int = N,
    sample_spacing: float = SAMPLE_SPACING,
    sgn: int = SGN,
    band: tuple[float, float] = BAND,
) -> np.ndarray:
    frequency = _sicd_frequency(n, sample_spacing, sgn)
    low, high = band
    inside = np.flatnonzero((frequency >= low) & (frequency <= high))
    return inside[np.argsort(frequency[inside])]


def _band_limited(n_columns: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    spectrum = generator.standard_normal((N, n_columns)) + 1j * generator.standard_normal((N, n_columns))
    spectrum *= rectangular_window(N, SAMPLE_SPACING, SGN, *BAND)[:, np.newaxis]
    return np.fft.ifft(spectrum, axis=0).astype(np.complex64)


def _hanning_formula(m: int) -> np.ndarray:
    samples = np.arange(m)
    return 0.5 - 0.5 * np.cos(2 * np.pi * samples / (m - 1))


def test_hanning_window_is_the_occupied_band() -> None:
    """Zero outside the band, symmetric, unit mean power, and the cosine formula."""

    window = range_hanning_window(N, SAMPLE_SPACING, SGN, *BAND)
    frequency = _sicd_frequency()
    inside = _inside()
    low, high = BAND
    outside = (frequency < low) | (frequency > high)

    assert np.all(window[outside] == 0), f"window is non-zero outside [{low}, {high}]"
    half = N // 2
    assert np.allclose(window[1:half], window[-1 : -half : -1]), "window is not symmetric about zero"
    mean_power = float(np.mean(window[inside] ** 2))
    assert abs(mean_power - 1.0) < 1e-12, f"mean w^2 on the band is {mean_power}, expected 1"

    formula = _hanning_formula(inside.size)
    scale = np.sqrt(inside.size / np.sum(formula ** 2))
    assert np.allclose(window[inside], formula * scale), (
        "window values do not follow 0.5 - 0.5 cos(2πn/(M-1)), scaled to unit mean power"
    )


def test_offset_band_uses_azimuth_frequency_sign() -> None:
    """With Sgn = -1, numpy's fftfreq is already the SICD frequency.

    A band from 0.10 to 0.20 cycles/m then covers positive fftfreq and leaves
    the negative mirror empty. The azimuth lag-1 centroid (-0.00155 cycles/m)
    has the same sign as DeltaKCOA (-0.00272). Multiplying fftfreq by Sgn
    reverses that sign.
    """

    n, spacing, sgn = 256, 1.0, -1
    low, high = 0.10, 0.20
    window = range_hanning_window(n, spacing, sgn, low, high)
    numpy_frequency = np.fft.fftfreq(n, d=spacing)
    negative = numpy_frequency < 0
    positive_band = (numpy_frequency >= low) & (numpy_frequency <= high)

    assert np.all(window[negative] == 0), "window is non-zero on negative fftfreq"
    assert np.max(window[positive_band]) > 0, "window is zero on the positive fftfreq band"

    def tone(frequency: float) -> np.ndarray:
        time = np.arange(n) * spacing
        return np.exp(2j * np.pi * frequency * time).astype(np.complex64)[:, np.newaxis]

    kept_frequency = numpy_frequency[positive_band][np.argmax(window[positive_band])]
    kept = apply_range_hanning(tone(kept_frequency), spacing, sgn, low, high)
    rejected = apply_range_hanning(tone(-kept_frequency), spacing, sgn, low, high)
    kept_power = float(np.mean(np.abs(kept) ** 2))
    rejected_power = float(np.mean(np.abs(rejected) ** 2))
    assert kept_power > 0.2, f"in-band tone power fell to {kept_power}"
    assert rejected_power < 1e-6, f"out-of-band tone power is {rejected_power}"


def test_mid_array_point_target_stays_in_place() -> None:
    """Filtering must not move a target that does not sit on sample 0."""

    index = N // 2
    rectangular = rectangular_window(N, SAMPLE_SPACING, SGN, *BAND)
    shift = np.exp(-2j * np.pi * np.fft.fftfreq(N) * index)
    spike = np.fft.ifft(rectangular * shift).astype(np.complex64)
    filtered = apply_range_hanning(spike[:, np.newaxis], SAMPLE_SPACING, SGN, *BAND)[:, 0]
    spike_peak = int(np.argmax(np.abs(spike)))
    filtered_peak = int(np.argmax(np.abs(filtered)))

    assert spike_peak == index, f"synthetic peak is at sample {spike_peak}, expected {index}"
    assert filtered_peak == spike_peak, f"filtered peak moved from {spike_peak} to {filtered_peak}"


def test_distributed_power_is_preserved() -> None:
    """Band-limited noise keeps its mean power, to within a few percent."""

    noise = _band_limited(n_columns=64, seed=0)
    filtered = apply_range_hanning(noise, SAMPLE_SPACING, SGN, *BAND)
    ratio = float(np.mean(np.abs(filtered) ** 2) / np.mean(np.abs(noise) ** 2))
    assert abs(ratio - 1.0) < 0.05, f"output/input power is {ratio:.4f}"


def test_rectangular_window_leaves_band_limited_data_unchanged() -> None:
    """On data that already lives in the band, the rectangular window is the identity."""

    data = _band_limited(n_columns=4, seed=1)
    window = rectangular_window(N, SAMPLE_SPACING, SGN, *BAND)
    filtered = apply_range_window(data, window)
    mismatch = float(np.max(np.abs(filtered - data)))
    assert np.allclose(filtered, data, rtol=1e-5, atol=1e-6), (
        f"rectangular window changed band-limited data by {mismatch}"
    )


def test_shape_and_dtype_are_preserved() -> None:
    data = _band_limited(n_columns=3, seed=2)
    filtered = apply_range_hanning(data, SAMPLE_SPACING, SGN, *BAND)
    assert filtered.shape == data.shape, f"shape {filtered.shape} != {data.shape}"
    assert filtered.dtype == data.dtype, f"dtype {filtered.dtype} != {data.dtype}"


def test_multilook_4x4() -> None:
    """2 by 2 looks on a 4 by 4 array, checked against the four block means."""

    power = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=float,
    )
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    looked = multilook(power, 2, 2)
    assert np.allclose(looked, expected), f"multilook returned\n{looked}\nexpected\n{expected}"


def run_tests() -> bool:
    """Execute every check and print whether it passed."""

    tests = [
        test_hanning_window_is_the_occupied_band,
        test_offset_band_uses_azimuth_frequency_sign,
        test_mid_array_point_target_stays_in_place,
        test_distributed_power_is_preserved,
        test_rectangular_window_leaves_band_limited_data_unchanged,
        test_shape_and_dtype_are_preserved,
        test_multilook_4x4,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as error:
            failed += 1
            reason = str(error).strip() or error.__class__.__name__
            print(f"FAIL {test.__name__}: {reason}")
        else:
            print(f"pass {test.__name__}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
