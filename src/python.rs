use pyo3::prelude::*;
use std::path::PathBuf;

#[pyfunction]
#[pyo3(signature = (input, output, input_format="nifti", bval=None, bvec=None, model="basic"))]
fn convert(
    input: PathBuf,
    output: PathBuf,
    input_format: &str,
    bval: Option<PathBuf>,
    bvec: Option<PathBuf>,
    model: &str,
) -> PyResult<()> {
    let format = match input_format.to_ascii_lowercase().as_str() {
        "nifti" => crate::InputFormat::Nifti,
        "dicom" => crate::InputFormat::Dicom,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "input_format must be 'nifti' or 'dicom'",
            ));
        }
    };
    let model = match model.to_ascii_lowercase().as_str() {
        "basic" => crate::Model::Basic,
        "auto" => crate::Model::Auto,
        "ivim" => crate::Model::Ivim,
        "dki" => crate::Model::Dki,
        "stretched-exponential" | "stretched_exponential" => crate::Model::StretchedExponential,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must be 'basic', 'auto', 'ivim', 'dki', or 'stretched-exponential'",
            ));
        }
    };
    crate::convert_with_model(
        &input,
        &output,
        format,
        bval.as_deref(),
        bvec.as_deref(),
        model,
    )
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert, module)?)?;
    Ok(())
}
