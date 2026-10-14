use anyhow::{Context, Result, bail};
use ndarray::{ArrayD, IxDyn};
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "python")]
mod python;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    Nifti,
    Dicom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Nifti,
    Dicom,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Model {
    #[default]
    Basic,
    Auto,
    Ivim,
    Dki,
    StretchedExponential,
}

impl Model {
    pub fn resolve(self, b_values: &[f64]) -> Result<Self> {
        let distinct = b_values
            .iter()
            .map(|b| (b * 10.0).round() as i64)
            .collect::<std::collections::HashSet<_>>();
        let resolved = match self {
            Self::Auto
                if b_values.iter().copied().fold(0.0, f64::max) > 1500.0
                    && distinct.len() >= 4
                    && b_values.iter().filter(|&&b| b > 0.0).count() >= 3 =>
            {
                Self::Dki
            }
            Self::Auto
                if b_values
                    .iter()
                    .filter(|&&b| b > 0.0 && b <= 200.0)
                    .map(|b| (*b * 10.0).round() as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    >= 2
                    && b_values
                        .iter()
                        .filter(|&&b| b > 200.0)
                        .map(|b| (*b * 10.0).round() as i64)
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        >= 2
                    && distinct.len() >= 4 =>
            {
                Self::Ivim
            }
            Self::Auto => Self::Basic,
            model => model,
        };
        validate_model(resolved, b_values)?;
        Ok(resolved)
    }
}

pub struct AdcMap {
    pub dimensions: Vec<usize>,
    pub values: Vec<f32>,
    pub header: Option<NiftiHeader>,
    pub source_dicom_slices: Option<Vec<PathBuf>>,
}

pub fn parse_vector_file(path: &Path) -> Result<Vec<f64>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let values = text
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .with_context(|| format!("invalid numeric value '{token}' in {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("{} contains no values", path.display());
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!(
            "{} must contain finite, non-negative values",
            path.display()
        );
    }
    Ok(values)
}

pub fn fit_adc(
    dimensions: &[usize],
    data: &[f32],
    b_values: &[f64],
    axis: usize,
) -> Result<Vec<f32>> {
    fit_model(dimensions, data, b_values, axis, Model::Basic)
}

pub fn fit_model(
    dimensions: &[usize],
    data: &[f32],
    b_values: &[f64],
    axis: usize,
    model: Model,
) -> Result<Vec<f32>> {
    if dimensions.len() < 2 || axis >= dimensions.len() {
        bail!("invalid volume dimensions or diffusion axis");
    }
    if dimensions[axis] != b_values.len() {
        bail!(
            "{} volumes but {} b-values",
            dimensions[axis],
            b_values.len()
        );
    }
    if dimensions.iter().product::<usize>() != data.len() {
        bail!("volume dimensions do not match image data");
    }
    if b_values.iter().any(|b| !b.is_finite() || *b < 0.0)
        || !b_values.contains(&0.0)
        || !b_values.iter().any(|b| *b > 0.0)
    {
        bail!("model fitting requires finite non-negative b-values including b=0 and b>0");
    }
    let model = model.resolve(b_values)?;
    let mut output_dimensions = dimensions.to_vec();
    output_dimensions.remove(axis);
    let mut output = vec![0.0; output_dimensions.iter().product()];
    let baseline_indices = b_values
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| (b == 0.0).then_some(i))
        .collect::<Vec<_>>();
    for (voxel, result) in output.iter_mut().enumerate() {
        let mut remainder = voxel;
        let mut coordinates = vec![0; dimensions.len()];
        for dimension in (0..dimensions.len()).rev() {
            if dimension != axis {
                coordinates[dimension] = remainder % dimensions[dimension];
                remainder /= dimensions[dimension];
            }
        }
        let mut signals = Vec::with_capacity(b_values.len());
        for (index, &b) in b_values.iter().enumerate() {
            coordinates[axis] = index;
            let flat = coordinates
                .iter()
                .zip(dimensions)
                .fold(0usize, |flat, (&coordinate, &size)| {
                    flat * size + coordinate
                });
            signals.push((b, data[flat] as f64));
        }
        let baseline_values = baseline_indices
            .iter()
            .filter_map(|&index| {
                let (_, signal) = signals[index];
                (signal.is_finite() && signal > 0.0).then_some(signal)
            })
            .collect::<Vec<_>>();
        let Some(baseline) = (!baseline_values.is_empty())
            .then(|| baseline_values.iter().sum::<f64>() / baseline_values.len() as f64)
        else {
            continue;
        };
        let samples = signals
            .iter()
            .filter_map(|&(b, signal)| {
                (b > 0.0 && signal.is_finite() && signal > 0.0)
                    .then_some((b, (signal / baseline).ln()))
            })
            .collect::<Vec<_>>();
        let fitted = match model {
            Model::Basic => fit_mono(&samples),
            Model::Dki => fit_dki(&samples),
            Model::Ivim => fit_ivim(&signals, baseline),
            Model::StretchedExponential => fit_stretched(&samples),
            Model::Auto => unreachable!(),
        };
        if let Some(value) = fitted.filter(|value| value.is_finite() && *value > 0.0) {
            *result = value as f32;
        }
    }
    Ok(output)
}

fn fit_mono(samples: &[(f64, f64)]) -> Option<f64> {
    let (sum_bb, sum_by) = samples
        .iter()
        .fold((0.0, 0.0), |(bb, by), &(b, y)| (bb + b * b, by + b * y));
    (sum_bb > f64::EPSILON).then_some(-sum_by / sum_bb)
}

fn fit_dki(samples: &[(f64, f64)]) -> Option<f64> {
    if samples.len() < 3 {
        return None;
    }
    let (mut a, mut b, mut c, mut d, mut e, mut y0, mut y1, mut y2) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y) in samples {
        let x2 = x * x;
        a += x2;
        b += x * x2;
        c += x2 * x2;
        d += x;
        e += x2;
        y0 += x * y;
        y1 += x2 * y;
        y2 += y;
    }
    let n = samples.len() as f64;
    let det = a * (c * n - e * e) - b * (b * n - e * d) + d * (b * e - c * d);
    if det.abs() < f64::EPSILON {
        return None;
    }
    let det_slope = y0 * (c * n - e * e) - b * (y1 * n - e * y2) + d * (y1 * e - c * y2);
    let slope = det_slope / det;
    let det_quad = a * (y1 * n - e * y2) - y0 * (b * n - e * d) + d * (b * y2 - y1 * d);
    let quadratic = det_quad / det;
    let diffusion = -slope;
    let kurtosis = 6.0 * quadratic / (diffusion * diffusion);
    (diffusion > 0.0 && kurtosis.is_finite() && (0.0..=3.0).contains(&kurtosis))
        .then_some(diffusion)
}

fn fit_ivim(signals: &[(f64, f64)], baseline: f64) -> Option<f64> {
    let samples = signals
        .iter()
        .filter_map(|&(b, signal)| {
            (b > 0.0 && signal.is_finite() && signal > 0.0).then_some((b, signal / baseline))
        })
        .collect::<Vec<_>>();
    if samples
        .iter()
        .filter(|(b, _)| *b <= 200.0)
        .map(|(b, _)| (*b * 10.0).round() as i64)
        .collect::<std::collections::HashSet<_>>()
        .len()
        < 2
        || samples
            .iter()
            .filter(|(b, _)| *b > 200.0)
            .map(|(b, _)| (*b * 10.0).round() as i64)
            .collect::<std::collections::HashSet<_>>()
            .len()
            < 2
    {
        return None;
    }
    let high = samples
        .iter()
        .filter(|(b, _)| *b > 200.0)
        .map(|(b, s)| (*b, s.ln()))
        .collect::<Vec<_>>();
    let mut best = None;
    let mut best_error = f64::INFINITY;
    for dstar in [0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3] {
        for fraction_step in 1..=40 {
            let fraction = fraction_step as f64 / 100.0;
            let adjusted = high
                .iter()
                .filter_map(|&(b, signal)| {
                    let tissue_signal =
                        (signal.exp() - fraction * (-b * dstar).exp()) / (1.0 - fraction);
                    (tissue_signal.is_finite() && tissue_signal > 0.0)
                        .then_some((b, tissue_signal.ln()))
                })
                .collect::<Vec<_>>();
            let Some(diffusion) = fit_mono(&adjusted).filter(|d| (0.0..=0.005).contains(d)) else {
                continue;
            };
            let error = samples
                .iter()
                .map(|&(b, observed)| {
                    let predicted =
                        (1.0 - fraction) * (-b * diffusion).exp() + fraction * (-b * dstar).exp();
                    (observed - predicted).powi(2)
                })
                .sum::<f64>();
            if error < best_error {
                best_error = error;
                best = Some(diffusion);
            }
        }
    }
    best
}

fn fit_stretched(samples: &[(f64, f64)]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let mut best = None;
    let mut best_error = f64::INFINITY;
    for step in 30..=100 {
        let alpha = step as f64 / 100.0;
        let numerator = samples
            .iter()
            .map(|&(b, y)| b.powf(alpha) * -y)
            .sum::<f64>();
        let denominator = samples
            .iter()
            .map(|&(b, _)| b.powf(2.0 * alpha))
            .sum::<f64>();
        if denominator <= f64::EPSILON {
            continue;
        }
        let coefficient = numerator / denominator;
        if !coefficient.is_finite() || coefficient <= 0.0 {
            continue;
        }
        let diffusion = coefficient.powf(1.0 / alpha);
        let error = samples
            .iter()
            .map(|&(b, y)| (y + coefficient * b.powf(alpha)).powi(2))
            .sum::<f64>();
        if diffusion.is_finite() && error < best_error {
            best_error = error;
            best = Some(diffusion);
        }
    }
    best
}

fn validate_model(model: Model, b_values: &[f64]) -> Result<()> {
    let distinct = b_values
        .iter()
        .map(|b| (b * 10.0).round() as i64)
        .collect::<std::collections::HashSet<_>>();
    let distinct_low = b_values
        .iter()
        .filter(|&&b| b > 0.0 && b <= 200.0)
        .map(|b| (*b * 10.0).round() as i64)
        .collect::<std::collections::HashSet<_>>();
    match model {
        Model::Ivim
            if distinct_low.len() < 2
                || b_values
                    .iter()
                    .filter(|&&b| b > 200.0)
                    .map(|b| (*b * 10.0).round() as i64)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    < 2 =>
        {
            bail!(
                "IVIM requires at least two distinct low b-values (≤200) and two distinct higher b-values"
            );
        }
        Model::Dki
            if distinct.len() < 4
                || b_values.iter().copied().fold(0.0, f64::max) <= 1500.0
                || b_values.iter().filter(|&&b| b > 0.0).count() < 3 =>
        {
            bail!(
                "DKI requires at least four distinct b-values and a maximum b-value of 1500 or higher"
            );
        }
        Model::StretchedExponential if distinct.len() < 3 => {
            bail!("stretched-exponential fitting requires at least three distinct b-values");
        }
        _ => {}
    }
    Ok(())
}

fn read_nifti(path: &Path) -> Result<AdcMap> {
    let object = ReaderOptions::new()
        .read_file(path)
        .with_context(|| format!("reading NIfTI {}", path.display()))?;
    let header = object.header().clone();
    let volume = object.into_volume().into_ndarray::<f32>()?;
    let dimensions = volume.shape().to_vec();
    let data = volume.iter().copied().collect::<Vec<_>>();
    Ok(AdcMap {
        dimensions,
        values: data,
        header: Some(header),
        source_dicom_slices: None,
    })
}

pub fn validate_bvecs(path: &Path, expected: usize) -> Result<()> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| {
                    token.parse::<f64>().with_context(|| {
                        format!("invalid vector value '{token}' in {}", path.display())
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.len() != 3
        || rows
            .iter()
            .any(|row| row.len() != expected || row.iter().any(|value| !value.is_finite()))
    {
        bail!(
            "{} must contain three rows of {expected} finite values",
            path.display()
        );
    }
    Ok(())
}

pub fn load_nifti_adc(input: &Path, bvals: &Path, bvecs: Option<&Path>) -> Result<AdcMap> {
    load_nifti_model(input, bvals, bvecs, Model::Basic)
}

pub fn load_nifti_model(
    input: &Path,
    bvals: &Path,
    bvecs: Option<&Path>,
    model: Model,
) -> Result<AdcMap> {
    let b_values = parse_vector_file(bvals)?;
    if let Some(bvecs) = bvecs {
        validate_bvecs(bvecs, b_values.len())?;
    }
    let mut map = read_nifti(input)?;
    if map.dimensions.len() < 4 {
        bail!("DWI NIfTI must have at least four dimensions");
    }
    let axis = map.dimensions.len() - 1;
    let values = fit_model(&map.dimensions, &map.values, &b_values, axis, model)?;
    map.dimensions.remove(axis);
    map.values = values;
    Ok(map)
}

fn dicom_tag_value(object: &dicom_object::DefaultDicomObject, name: &str) -> Result<String> {
    Ok(object.element_by_name(name)?.to_str()?.to_string())
}

fn element_b_value(element: &dicom_object::mem::InMemElement) -> Option<f64> {
    let value = element.to_float64().ok()?;
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn dicom_b_value(object: &dicom_object::DefaultDicomObject) -> Option<f64> {
    use dicom_core::Tag;

    if let Ok(element) = object.element(Tag(0x0018, 0x9117))
        && let Some(sequence) = element.items()
    {
        for item in sequence {
            if let Some(value) = item
                .element(Tag(0x0018, 0x9087))
                .ok()
                .and_then(element_b_value)
            {
                return Some(value);
            }
        }
    }

    for tag in [
        Tag(0x0018, 0x9087),
        Tag(0x0019, 0x100C),
        Tag(0x0019, 0x10CC),
        Tag(0x2001, 0x1003),
        Tag(0x0029, 0x1014),
        Tag(0x0065, 0x1009),
    ] {
        if let Ok(element) = object.element(tag)
            && let Some(value) = element_b_value(element)
        {
            return Some(value);
        }
    }

    if let Ok(element) = object.element(Tag(0x0043, 0x1039))
        && let Ok(values) = element.to_multi_float64()
        && let Some(&encoded_b_value) = values.first()
    {
        let b_value = encoded_b_value.rem_euclid(1_000_000_000.0);
        if b_value.is_finite() && b_value >= 0.0 {
            return Some(b_value);
        }
    }

    for tag in [Tag(0x0029, 0x1010), Tag(0x0029, 0x1020)] {
        if let Ok(element) = object.element(tag) {
            let Ok(bytes) = element.value().to_bytes() else {
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            if let Some(value) = parse_csa_b_value(&text) {
                return Some(value);
            }
        }
    }

    None
}

pub fn parse_csa_b_value(text: &str) -> Option<f64> {
    let lines = text.lines().collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        let marker = line.trim();
        if marker.eq_ignore_ascii_case("B_value")
            || marker.starts_with("B_value ")
            || marker.starts_with("B_value=")
        {
            if let Some((_, value)) = marker.split_once('=')
                && let Ok(parsed) = value.trim().parse::<f64>()
                && parsed.is_finite()
                && parsed >= 0.0
            {
                return Some(parsed);
            }
            let mut numeric_values = Vec::new();
            for value_line in lines.iter().skip(index + 1).take(6) {
                if value_line.trim().is_empty() {
                    continue;
                }
                if value_line.trim().starts_with("B_value") {
                    break;
                }
                let values = value_line
                    .split_once('=')
                    .map_or(*value_line, |(_, values)| values);
                numeric_values.extend(
                    values
                        .split(|character: char| {
                            !character.is_ascii_digit() && character != '.' && character != '-'
                        })
                        .filter(|token| !token.is_empty())
                        .filter_map(|token| token.parse::<f64>().ok())
                        .filter(|value| value.is_finite() && *value >= 0.0),
                );
            }
            if let Some(value) = numeric_values.get(2) {
                return Some(*value);
            }
            if let Some(value) = numeric_values.last() {
                return Some(*value);
            }
        }
        if let Some((key, value)) = line.split_once('=')
            && key.trim().eq_ignore_ascii_case("B_value")
            && let Ok(parsed) = value.trim().parse::<f64>()
            && parsed.is_finite()
            && parsed >= 0.0
        {
            return Some(parsed);
        }
    }
    None
}

type DicomImage = (Vec<f32>, PathBuf);
type DicomVolume = Vec<Vec<DicomImage>>;

fn read_dicom_series(
    directory: &Path,
    supplied_bvals: Option<&[f64]>,
    model: Model,
) -> Result<AdcMap> {
    use dicom_object::open_file;
    use dicom_pixeldata::PixelDecoder;
    let mut paths = fs::read_dir(directory)
        .with_context(|| format!("reading DICOM directory {}", directory.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    paths.sort();
    let mut slices = Vec::new();
    let mut series_uid: Option<String> = None;
    for path in paths {
        let object = match open_file(&path) {
            Ok(object) => object,
            Err(_) => continue,
        };
        let pixels = match object.decode_pixel_data() {
            Ok(pixels) => pixels,
            Err(_) => continue,
        };
        let dimensions = [
            pixels.number_of_frames() as usize,
            pixels.rows() as usize,
            pixels.columns() as usize,
            pixels.samples_per_pixel() as usize,
        ];
        if dimensions[0] != 1 || dimensions[3] != 1 {
            bail!("{}: expected one monochrome DICOM frame", path.display());
        }
        let current_series = dicom_tag_value(&object, "SeriesInstanceUID")?;
        if let Some(expected) = &series_uid {
            if expected != &current_series {
                continue;
            }
        } else {
            series_uid = Some(current_series);
        }
        let rows = dimensions[1];
        let columns = dimensions[2];
        let data = pixels.to_vec::<f32>()?;
        let b = if dimensions[0] == 1 {
            dicom_b_value(&object)
        } else {
            None
        };
        let position = dicom_tag_value(&object, "ImagePositionPatient")?
            .split('\\')
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if position.len() != 3 {
            bail!("{}: invalid ImagePositionPatient", path.display());
        }
        let orientation = dicom_tag_value(&object, "ImageOrientationPatient")?
            .split('\\')
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if orientation.len() != 6 {
            bail!("{}: invalid ImageOrientationPatient", path.display());
        }
        let normal = [
            orientation[1] * orientation[5] - orientation[2] * orientation[4],
            orientation[2] * orientation[3] - orientation[0] * orientation[5],
            orientation[0] * orientation[4] - orientation[1] * orientation[3],
        ];
        let location = position.iter().zip(normal).map(|(p, n)| p * n).sum::<f64>();
        let acquisition = dicom_tag_value(&object, "AcquisitionNumber")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let instance = dicom_tag_value(&object, "InstanceNumber")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(slices.len() + 1);
        slices.push((
            location,
            b,
            acquisition,
            instance,
            data,
            path,
            rows,
            columns,
        ));
    }
    if slices.is_empty() {
        bail!(
            "no readable single-frame DICOM images found in {}",
            directory.display()
        );
    }
    slices.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.3.cmp(&right.3)));
    if slices
        .iter()
        .any(|slice| slice.6 != slices[0].6 || slice.7 != slices[0].7)
    {
        bail!("DICOM slices have inconsistent dimensions");
    }
    let (rows, columns) = (slices[0].6, slices[0].7);
    let image_len = rows * columns;
    if slices.iter().any(|slice| slice.1.is_none()) && supplied_bvals.is_none() {
        bail!(
            "Could not find diffusion b-values in the supported standard, vendor-private, or Siemens CSA DICOM tags. Provide a manual .bval file with --bval, and please open an issue at https://github.com/mjbarrett/dwi-to-adc/issues with the scanner vendor/model and a de-identified example."
        );
    }
    if let Some(values) = supplied_bvals {
        if values.len() == slices.len() {
            for (slice, &b) in slices.iter_mut().zip(values) {
                slice.1 = Some(b);
            }
        } else {
            let mut acquisition_ids = slices
                .iter()
                .filter_map(|slice| slice.2)
                .collect::<Vec<_>>();
            acquisition_ids.sort_unstable();
            acquisition_ids.dedup();
            if acquisition_ids.len() == values.len() {
                for slice in &mut slices {
                    slice.1 = slice
                        .2
                        .and_then(|id| {
                            acquisition_ids
                                .iter()
                                .position(|candidate| *candidate == id)
                        })
                        .map(|index| values[index]);
                }
            } else {
                let unique_locations = slices.iter().map(|slice| slice.0).fold(
                    Vec::new(),
                    |mut locations, location| {
                        if locations
                            .last()
                            .is_none_or(|last: &f64| (location - *last).abs() >= 0.01)
                        {
                            locations.push(location);
                        }
                        locations
                    },
                );
                let images_per_position = slices.len() / unique_locations.len();
                if slices.len() % unique_locations.len() != 0 || values.len() != images_per_position
                {
                    bail!(
                        "{} DICOM images occupy {} slice positions with {} images per position; {} b-values cannot be mapped",
                        slices.len(),
                        unique_locations.len(),
                        images_per_position,
                        values.len()
                    );
                }
                for (position_index, location) in unique_locations.iter().enumerate() {
                    let first = slices
                        .iter()
                        .position(|slice| (slice.0 - *location).abs() < 0.01)
                        .unwrap();
                    for (volume_index, slice) in slices[first..first + images_per_position]
                        .iter_mut()
                        .enumerate()
                    {
                        slice.1 = Some(values[volume_index]);
                    }
                    if position_index + 1 < unique_locations.len() {
                        let next = slices[first + images_per_position].0;
                        if (next - *location).abs() < 0.01 {
                            bail!("DICOM slices are not grouped by position");
                        }
                    }
                }
            }
        }
    }
    let mut b_values = slices
        .iter()
        .map(|slice| slice.1.unwrap())
        .collect::<Vec<_>>();
    b_values.sort_by(f64::total_cmp);
    b_values.dedup_by(|left, right| (*left - *right).abs() < 0.5);
    if !b_values.contains(&0.0) || b_values.iter().all(|b| *b == 0.0) {
        bail!("DICOM series needs at least one b=0 and one non-zero diffusion image");
    }
    let mut positions: Vec<(f64, DicomVolume)> = Vec::new();
    for (location, b, _, _, pixels, path, _, _) in slices {
        let b_index = b_values
            .iter()
            .position(|candidate| (candidate - b.unwrap()).abs() < 0.5)
            .context("invalid DICOM b-value")?;
        let Some((_, acquisitions)) = positions
            .last_mut()
            .filter(|(last, _)| (location - *last).abs() < 0.01)
        else {
            positions.push((location, vec![vec![]; b_values.len()]));
            positions.last_mut().unwrap().1[b_index].push((pixels, path));
            continue;
        };
        acquisitions[b_index].push((pixels, path));
    }
    let mut data = vec![0.0; positions.len() * image_len * b_values.len()];
    let mut source_slices = Vec::with_capacity(positions.len());
    for (z, (_, acquisitions)) in positions.iter().enumerate() {
        if acquisitions.iter().any(Vec::is_empty) {
            bail!("DICOM diffusion volumes do not contain matching slices at every position");
        }
        let template = acquisitions[0]
            .first()
            .context("missing DICOM template slice")?;
        source_slices.push(template.1.clone());
        for (b_index, acquisitions_at_b) in acquisitions.iter().enumerate() {
            for pixel in 0..image_len {
                let mean = acquisitions_at_b
                    .iter()
                    .map(|(values, _)| values[pixel] as f64)
                    .sum::<f64>()
                    / acquisitions_at_b.len() as f64;
                data[(z * image_len + pixel) * b_values.len() + b_index] = mean as f32;
            }
        }
    }
    let values = fit_model(
        &[positions.len(), rows, columns, b_values.len()],
        &data,
        &b_values,
        3,
        model,
    )?;
    Ok(AdcMap {
        dimensions: vec![positions.len(), rows, columns],
        values,
        header: None,
        source_dicom_slices: Some(source_slices),
    })
}

pub fn convert(
    input: &Path,
    output: &Path,
    format: InputFormat,
    bvals: Option<&Path>,
    bvecs: Option<&Path>,
) -> Result<()> {
    convert_with_model(input, output, format, bvals, bvecs, Model::Basic)
}

pub fn convert_with_model(
    input: &Path,
    output: &Path,
    format: InputFormat,
    bvals: Option<&Path>,
    bvecs: Option<&Path>,
    model: Model,
) -> Result<()> {
    let mut map = match format {
        InputFormat::Nifti => load_nifti_model(
            input,
            bvals.context("NIfTI input requires --bval")?,
            bvecs,
            model,
        )?,
        InputFormat::Dicom => {
            let parsed_bvals = bvals.map(parse_vector_file).transpose()?;
            read_dicom_series(input, parsed_bvals.as_deref(), model)?
        }
    };
    match output.extension().and_then(|extension| extension.to_str()) {
        Some("nii" | "gz") => save_nifti(output, &map)?,
        _ => save_dicom_series(output, &mut map)?,
    }
    Ok(())
}

pub fn save_nifti(path: &Path, map: &AdcMap) -> Result<()> {
    let array = ArrayD::from_shape_vec(IxDyn(&map.dimensions), map.values.clone())?;
    let mut writer = WriterOptions::new(path);
    if let Some(header) = &map.header {
        writer = writer.reference_header(header);
    }
    writer
        .write_nifti(&array)
        .with_context(|| format!("writing NIfTI {}", path.display()))
}

pub fn save_dicom_series(directory: &Path, map: &mut AdcMap) -> Result<()> {
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::tags;
    use dicom_object::open_file;
    if directory.exists() {
        bail!("output directory {} already exists", directory.display());
    }
    fs::create_dir_all(directory)?;
    let sources = map
        .source_dicom_slices
        .as_ref()
        .context("DICOM output requires DICOM source images")?;
    let mut templates = sources
        .iter()
        .map(open_file)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if templates.len() != map.dimensions[0] {
        bail!("DICOM output needs one source slice for each ADC slice");
    }
    let rows = templates[0].element_by_name("Rows")?.to_int::<u16>()? as usize;
    let columns = templates[0].element_by_name("Columns")?.to_int::<u16>()? as usize;
    if map.dimensions.len() != 3 || map.dimensions[1..] != [rows, columns] {
        bail!("ADC dimensions do not match DICOM source slices");
    }
    let slice_count = map.dimensions[0];
    let maximum = map
        .values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f32, f32::max);
    let scale = if maximum.is_finite() && maximum > 0.0 {
        maximum / 32767.0
    } else {
        1.0
    };
    let series_uid = format!("2.25.{}", uuid::Uuid::new_v4().as_u128());
    for index in 0..slice_count {
        let mut slice = templates.remove(0);
        let start = index * rows * columns;
        let pixels = map.values[start..start + rows * columns]
            .iter()
            .map(|value| (value / scale).round().clamp(0.0, 32767.0) as u16)
            .collect::<Vec<_>>();
        let bytes = pixels
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        slice.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OW,
            PrimitiveValue::U8(bytes.into()),
        ));
        slice.put(DataElement::new(
            tags::ROWS,
            VR::US,
            PrimitiveValue::U16(vec![rows as u16].into()),
        ));
        slice.put(DataElement::new(
            tags::COLUMNS,
            VR::US,
            PrimitiveValue::U16(vec![columns as u16].into()),
        ));
        slice.put(DataElement::new(
            tags::BITS_ALLOCATED,
            VR::US,
            PrimitiveValue::U16(vec![16].into()),
        ));
        slice.put(DataElement::new(
            tags::BITS_STORED,
            VR::US,
            PrimitiveValue::U16(vec![16].into()),
        ));
        slice.put(DataElement::new(
            tags::HIGH_BIT,
            VR::US,
            PrimitiveValue::U16(vec![15].into()),
        ));
        slice.put(DataElement::new(
            tags::PIXEL_REPRESENTATION,
            VR::US,
            PrimitiveValue::U16(vec![0].into()),
        ));
        slice.put(DataElement::new(
            tags::RESCALE_SLOPE,
            VR::DS,
            PrimitiveValue::from(scale.to_string()),
        ));
        slice.put(DataElement::new(
            tags::RESCALE_INTERCEPT,
            VR::DS,
            PrimitiveValue::from("0"),
        ));
        slice.put_str(tags::MODALITY, VR::CS, "OT");
        slice.put_str(tags::SERIES_DESCRIPTION, VR::LO, "ADC map (mm2/s)");
        slice.put_str(
            tags::DERIVATION_DESCRIPTION,
            VR::ST,
            "Scalar mono-exponential ADC fit; units mm2/s",
        );
        slice.put_str(tags::SERIES_INSTANCE_UID, VR::UI, &series_uid);
        slice.put_str(tags::SOP_CLASS_UID, VR::UI, "1.2.840.10008.5.1.4.1.1.4");
        slice.put_str(
            tags::SOP_INSTANCE_UID,
            VR::UI,
            format!("2.25.{}", uuid::Uuid::new_v4().as_u128()),
        );
        slice.put_str(tags::IMAGE_TYPE, VR::CS, "DERIVED\\\\SECONDARY\\\\ADC");
        for name in [
            "DiffusionBValue",
            "DiffusionBFactor",
            "DiffusionGradientDirection",
        ] {
            if slice.element_by_name(name).is_ok() {
                slice.remove_element_by_name(name)?;
            }
        }
        slice.put(DataElement::new(
            tags::INSTANCE_NUMBER,
            VR::IS,
            PrimitiveValue::from((index + 1).to_string()),
        ));
        slice.write_to_file(directory.join(format!("adc_{:04}.dcm", index + 1)))?;
    }
    Ok(())
}
