# dwi-to-adc

Rust command-line program and Python package for generating scalar apparent diffusion coefficient (ADC) maps from diffusion-weighted MRI.

## Supported data

- NIfTI DWI (`.nii` or `.nii.gz`) with matching FSL `.bval` and optional `.bvec` validation. The diffusion volumes are along the final NIfTI axis; vectors are validated but not used by scalar ADC fitting.
- A directory of single-frame DICOM images from one diffusion series. The reader checks standard diffusion sequences/tags, Siemens CSA headers, and known Siemens, GE, Philips, Canon/Toshiba, and UIH private b-value tags. If b-values cannot be extracted, provide a manual FSL `.bval` file with `--bval`; the tool reports this fallback and asks users to open an issue if the format is not supported. Multi-frame Enhanced DICOM is not yet supported.
- NIfTI ADC output (`.nii` or `.nii.gz`) or a directory of derived DICOM slices.

ADC is estimated by linear least squares of $\ln(S/S_0)$ versus b-value using all positive b-values and the mean available b=0 image. Non-positive/non-finite signals and non-decaying fits produce zero. ADC values are in mm²/s when input b-values use s/mm².

## Standalone binary

```sh
cargo build --release
./target/release/dwi-to-adc DWI.nii.gz adc.nii.gz --bval DWI.bval --bvec DWI.bvec
./target/release/dwi-to-adc ./dicom-series adc-dicom/ --input-format dicom
```

Published releases attach standalone archives for Linux x86_64 and Apple Silicon macOS. The `dwi-to-adc` command is also installed by the Python package. A minimal container image is published to `ghcr.io/<owner>/<repository>` for each release.

An input directory is inferred as DICOM; a file is inferred as NIfTI. DICOM output is selected by passing an output directory; NIfTI output by `.nii` or `.nii.gz` extension. DICOM output requires an empty, non-existing destination directory. DICOM pixel values are stored as unsigned 16-bit integers with rescale slope set to preserve ADC magnitude. Review derived DICOM metadata with your clinical imaging workflow before clinical use.

## Python package

```sh
python -m pip install lavlab-adc

dwi-to-adc DWI.nii.gz adc.nii.gz --bval DWI.bval
dwi-to-adc ./dicom-series adc-dicom --input-format dicom
```

On supported platforms, prebuilt wheels are published for Linux x86_64 and Apple Silicon macOS. The extension uses PyO3's stable ABI, so each wheel supports Python 3.9 and newer CPython versions on those platforms.

To build from source locally:

```sh
python -m pip install maturin
maturin develop --release
```

```python
import dwi_to_adc

dwi_to_adc.convert("DWI.nii.gz", "adc.nii.gz", bval="DWI.bval", bvec="DWI.bvec")
dwi_to_adc.convert("dicom-series", "adc-dicom", input_format="dicom")
```

## Development

```sh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
