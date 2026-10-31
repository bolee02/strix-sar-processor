# strix-sar-processor

Simple Single Look Complex (SLC) image processing for Synspective's StriX satellites.

## Setup

```bash
uv sync
```

## Metadata

Print shape, dtype, and `Grid.Row` / `Grid.Col` for an SLC:

```bash
uv run python processing/loader.py slcs/<file>.nitf
```

## Plots

Write the range spectrum, Hanning window, Busan harbor comparison, and power overview to `output/`:

```bash
uv run python processing/plotter.py slcs/<file>.nitf
```

## Tests

```bash
uv run python processing/tests/test_windowing.py
uv run pytest processing/tests
```
