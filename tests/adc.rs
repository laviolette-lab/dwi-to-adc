use dwi_to_adc::{Model, fit_adc, fit_model, parse_csa_b_value};

#[test]
fn parses_siemens_csa_b_value() {
    assert_eq!(
        parse_csa_b_value("B_value\n\t1\n\t0\n\t1000\n"),
        Some(1000.0)
    );
    assert_eq!(parse_csa_b_value("B_value = 0"), Some(0.0));
    assert_eq!(parse_csa_b_value("no diffusion metadata"), None);
}

#[test]
fn fits_adc_for_last_and_middle_axes() {
    let adc = 0.001_f32;
    let data = [
        1000.0,
        1000.0,
        (1000.0 * (-adc * 1000.0).exp()),
        (1000.0 * (-adc * 2000.0).exp()),
    ];
    let result = fit_adc(&[1, 1, 4], &data, &[0.0, 0.0, 1000.0, 2000.0], 2).unwrap();
    assert!((result[0] - adc).abs() < 1e-6);

    let data = [1000.0, data[2], data[3], 500.0, 250.0, 125.0];
    let result = fit_adc(&[2, 3], &data, &[0.0, 1000.0, 2000.0], 1).unwrap();
    assert!((result[0] - adc).abs() < 1e-6);
}

#[test]
fn returns_zero_for_nondecaying_or_invalid_signals() {
    let data = [10.0, 11.0, 0.0, f32::NAN];
    let result = fit_adc(&[1, 4], &data, &[0.0, 1000.0, 2000.0, 3000.0], 1).unwrap();
    assert_eq!(result, vec![0.0]);
}

#[test]
fn auto_selects_model_from_shell_coverage() {
    assert_eq!(
        Model::Auto
            .resolve(&[0.0, 50.0, 100.0, 500.0, 1000.0])
            .unwrap(),
        Model::Ivim
    );
    assert_eq!(
        Model::Auto.resolve(&[0.0, 500.0, 1000.0, 2000.0]).unwrap(),
        Model::Dki
    );
    assert_eq!(
        Model::Auto.resolve(&[0.0, 500.0, 1000.0]).unwrap(),
        Model::Basic
    );
}

#[test]
fn rejects_models_without_required_shells() {
    assert!(Model::Ivim.resolve(&[0.0, 100.0, 1000.0]).is_err());
    assert!(Model::Dki.resolve(&[0.0, 500.0, 1000.0, 1500.0]).is_err());
}

#[test]
fn fits_dki_and_stretched_exponential_diffusion() {
    let b_values = [0.0, 500.0, 1000.0, 2000.0];
    let d = 0.001_f64;
    let kurtosis = 0.8_f64;
    let dki_signal = b_values
        .iter()
        .map(|&b| (1000.0 * (-b * d + b * b * d * d * kurtosis / 6.0).exp()) as f32)
        .collect::<Vec<_>>();
    let dki = fit_model(&[1, 4], &dki_signal, &b_values, 1, Model::Dki).unwrap();
    assert!((dki[0] - d as f32).abs() < 1e-5);

    let stretched_signal = b_values
        .iter()
        .map(|&b| (1000.0 * (-(b * d).powf(0.8)).exp()) as f32)
        .collect::<Vec<_>>();
    let stretched = fit_model(
        &[1, 4],
        &stretched_signal,
        &b_values,
        1,
        Model::StretchedExponential,
    )
    .unwrap();
    assert!((stretched[0] - d as f32).abs() < 1e-4);
}

#[test]
fn fits_ivim_tissue_diffusion_coefficient() {
    let b_values = [0.0, 50.0, 100.0, 500.0, 1000.0];
    let d = 0.001_f64;
    let d_star = 0.02_f64;
    let fraction = 0.2_f64;
    let signal = b_values
        .iter()
        .map(|&b| {
            (1000.0 * ((1.0 - fraction) * (-b * d).exp() + fraction * (-b * d_star).exp())) as f32
        })
        .collect::<Vec<_>>();
    let fitted = fit_model(&[1, 5], &signal, &b_values, 1, Model::Ivim).unwrap();
    assert!((fitted[0] - d as f32).abs() < 2e-4);
}

#[test]
fn rejects_mismatched_gradient_count() {
    assert!(fit_adc(&[2, 2], &[1.0; 4], &[0.0, 1000.0, 2000.0], 1).is_err());
}
