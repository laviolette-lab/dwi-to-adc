"""Diffusion MRI to ADC conversion."""

import argparse
from pathlib import Path

from ._native import convert


def main() -> None:
    """Provide the ``dwi-to-adc`` command for Python package installs."""
    parser = argparse.ArgumentParser(description="Convert diffusion-weighted MRI data to ADC maps")
    parser.add_argument("input", type=Path, help="NIfTI file or directory containing DICOM slices")
    parser.add_argument("output", type=Path, help="NIfTI path or directory for DICOM slices")
    parser.add_argument("--input-format", choices=("nifti", "dicom"))
    parser.add_argument("--bval", type=Path, help="FSL .bval file")
    parser.add_argument("--bvec", type=Path, help="Optional FSL .bvec file")
    args = parser.parse_args()
    input_format = args.input_format or ("dicom" if args.input.is_dir() else "nifti")
    try:
        convert(args.input, args.output, input_format, args.bval, args.bvec)
    except Exception as error:
        parser.exit(1, f"dwi-to-adc: error: {error}\n")


__all__ = ["convert", "main"]
