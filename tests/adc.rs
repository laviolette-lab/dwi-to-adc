use dwi_to_adc::{fit_adc, parse_csa_b_value};

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
fn rejects_mismatched_gradient_count() {
    assert!(fit_adc(&[2, 2], &[1.0; 4], &[0.0, 1000.0, 2000.0], 1).is_err());
}
