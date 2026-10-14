use pyo3::prelude::*;
use std::path::PathBuf;

#[pyfunction]
#[pyo3(signature = (input, output, input_format="nifti", bval=None, bvec=None))]
fn convert(
    input: PathBuf,
    output: PathBuf,
    input_format: &str,
    bval: Option<PathBuf>,
    bvec: Option<PathBuf>,
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
    crate::convert(&input, &output, format, bval.as_deref(), bvec.as_deref())
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert, module)?)?;
    Ok(())
}
