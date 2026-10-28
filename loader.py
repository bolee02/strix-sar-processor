"""Open a StriX SICD SLC and report the array and grid metadata."""

import argparse
from pathlib import Path

import numpy as np
from sarpy.io.complex.base import SICDTypeReader
from sarpy.io.complex.converter import open_complex
from sarpy.io.complex.sicd_elements.Grid import DirParamType
from sarpy.io.complex.sicd_elements.SICD import SICDType

REPO_ROOT = Path(__file__).resolve().parent


def open_slc(path: str | Path) -> tuple[SICDTypeReader, SICDType]:
    """Open an SLC NITF file and its SICD metadata."""

    reader = open_complex(str(path))
    sicd = reader.get_sicds_as_tuple()[0]
    return reader, sicd


def slant_range_at_row(reader: SICDTypeReader, sicd: SICDType, row: int) -> float:
    """Slant range (m) of a data row."""

    # Slant range at SCP    
    r_scp = sicd.SCPCOA.SlantRange

    # Full row index
    full_row = row + sicd.ImageData.FirstRow

    # Slant range at row
    return r_scp + (full_row - sicd.ImageData.SCPPixel.Row) * sicd.Grid.Row.SS


def describe(reader: SICDTypeReader, sicd: SICDType, path: str | Path | None = None) -> None:
    """Print shape, dtype, and the `Grid.Row` / `Grid.Col` fields."""

    rows, cols = reader.get_data_size_as_tuple()[0]
    sample = np.asarray(reader[0:1, 0:1])
    if path is not None:
        print(path)
    print(f"shape: ({rows}, {cols})")
    print(f"dtype: {sample.dtype}")
    print(f"Grid.Type: {sicd.Grid.Type}")
    for name, direction in (("Grid.Row", sicd.Grid.Row), ("Grid.Col", sicd.Grid.Col)):
        print(name)
        print(f"  SS (m): {direction.SS}")
        print(f"  ImpRespBW (cyc/m): {direction.ImpRespBW:.4f}")
        print(f"  WgtType: {direction.WgtType.WindowName}")
        print(f"  DeltaKCOAPoly (cyc/m): {direction.DeltaKCOAPoly.Coefs}")
    near = slant_range_at_row(reader, sicd, 0)
    far = slant_range_at_row(reader, sicd, rows - 1)
    print(f"Row 0 slant range: {near:.1f} m, last row: {far:.1f} m ")

def main():
    parser = argparse.ArgumentParser(description="Print SLC grid metadata.")
    parser.add_argument("slc", type=Path, help="path to the SLC .nitf file")
    args = parser.parse_args()

    reader, sicd = open_slc(args.slc)
    describe(reader, sicd)


if __name__ == "__main__":
    main()
