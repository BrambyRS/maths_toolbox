pub mod auto_diff;
pub mod lin_alg;

pub fn glq_quadrature(n: usize) -> Vec<(f64, f64)> {
    match n {
        1 => vec![(0.0, 2.0)],
        2 => vec![(-1.0 / f64::sqrt(3.0), 1.0), (1.0 / f64::sqrt(3.0), 1.0)],
        3 => vec![
            (-f64::sqrt(3.0 / 5.0), 5.0 / 9.0),
            (0.0, 8.0 / 9.0),
            (f64::sqrt(3.0 / 5.0), 5.0 / 9.0),
        ],
        4 => vec![
            (
                -f64::sqrt((3.0 + 2.0 * f64::sqrt(6.0 / 5.0)) / 7.0),
                (18.0 - f64::sqrt(30.0)) / 36.0,
            ),
            (
                -f64::sqrt((3.0 - 2.0 * f64::sqrt(6.0 / 5.0)) / 7.0),
                (18.0 + f64::sqrt(30.0)) / 36.0,
            ),
            (
                f64::sqrt((3.0 - 2.0 * f64::sqrt(6.0 / 5.0)) / 7.0),
                (18.0 + f64::sqrt(30.0)) / 36.0,
            ),
            (
                f64::sqrt((3.0 + 2.0 * f64::sqrt(6.0 / 5.0)) / 7.0),
                (18.0 - f64::sqrt(30.0)) / 36.0,
            ),
        ],
        _ => panic!("Quadrature of order {} not implemented", n),
    }
}

pub fn glq_interval(a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    let mut glq_points = glq_quadrature(n);
    for i in 0..n {
        let (xi, wi) = glq_points[i];
        glq_points[i].0 = 0.5 * (b - a) * xi + 0.5 * (a + b);
        glq_points[i].1 = 0.5 * (b - a) * wi;
    }
    return glq_points;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glq_quadrature() {
        let result = glq_quadrature(2);
        assert_eq!(result.len(), 2);

        let func = |x: f64| -> f64 { x.powi(2) };
        let expected: f64 = 2.0 / 3.0; // Integral of x^2 from -1 to 1
        let mut integral: f64 = 0.0;
        for (xi, wi) in result {
            integral += wi * func(xi);
        }
        assert!((integral - expected).abs() < 1e-6);
    }

    #[test]
    fn test_glq_quadrature_order_3() {
        let result = glq_quadrature(3);
        assert_eq!(result.len(), 3);

        let func = |x: f64| -> f64 { x.powi(3) - 2.0 * x.powi(2) }; // Odd function
        let expected: f64 = -4.0 / 3.0; // Integral of x^3 - 2x^2 from -1 to 1
        let mut integral: f64 = 0.0;
        for (xi, wi) in result {
            integral += wi * func(xi);
        }
        assert!((integral - expected).abs() < 1e-6);
    }

    #[test]
    fn test_glq_quadrature_order_4() {
        let result = glq_quadrature(4);
        assert_eq!(result.len(), 4);

        let func = |x: f64| -> f64 { x.powi(4) };
        let expected: f64 = 2.0 / 5.0; // Integral of x^4 from -1 to 1
        let mut integral: f64 = 0.0;
        for (xi, wi) in result {
            integral += wi * func(xi);
        }
        assert!((integral - expected).abs() < 1e-6);
    }

    #[test]
    fn test_glq_interval() {
        let a: f64 = 0.0;
        let b: f64 = 1.0;
        let result = glq_interval(a, b, 2);
        assert_eq!(result.len(), 2);

        let func = |x: f64| -> f64 { x.powi(2) };
        let expected: f64 = 1.0 / 3.0; // Integral of x^2 from 0 to 1
        let mut integral: f64 = 0.0;
        for (xi, wi) in result {
            integral += wi * func(xi);
        }
        assert!((integral - expected).abs() < 1e-6);
    }
}
