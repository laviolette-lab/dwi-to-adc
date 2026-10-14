use clap::{Parser, ValueEnum};
use dwi_to_adc::{InputFormat, Model, convert_with_model};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Format {
    Nifti,
    Dicom,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelArg {
    Basic,
    Auto,
    Ivim,
    Dki,
    StretchedExponential,
}

impl From<ModelArg> for Model {
    fn from(model: ModelArg) -> Self {
        match model {
            ModelArg::Basic => Self::Basic,
            ModelArg::Auto => Self::Auto,
            ModelArg::Ivim => Self::Ivim,
            ModelArg::Dki => Self::Dki,
            ModelArg::StretchedExponential => Self::StretchedExponential,
        }
    }
}

#[derive(Debug, Parser)]
#[command(version, about = "Convert diffusion-weighted MRI data to ADC maps")]
struct Args {
    /// Input NIfTI file or directory containing DICOM slices
    input: PathBuf,
    /// Output .nii/.nii.gz path or directory for DICOM slices
    output: PathBuf,
    /// Input format (inferred from input path when omitted)
    #[arg(long, value_enum)]
    input_format: Option<Format>,
    /// FSL .bval file (required for NIfTI; manual fallback for unsupported DICOM tags)
    #[arg(long)]
    bval: Option<PathBuf>,
    /// Optional FSL b-vectors file; validated but not used by scalar fitting
    #[arg(long)]
    bvec: Option<PathBuf>,
    /// Fitting model (default: basic)
    #[arg(long, value_enum, default_value_t = ModelArg::Basic)]
    model: ModelArg,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let format = args.input_format.unwrap_or_else(|| {
        if args.input.is_dir() {
            Format::Dicom
        } else {
            Format::Nifti
        }
    });
    let format = match format {
        Format::Nifti => InputFormat::Nifti,
        Format::Dicom => InputFormat::Dicom,
    };
    convert_with_model(
        &args.input,
        &args.output,
        format,
        args.bval.as_deref(),
        args.bvec.as_deref(),
        args.model.into(),
    )?;
    Ok(())
}
