/* Automatic differentiation type */

pub struct DiffNum<T> {
    pub f: T,
    pub df: T,
}

// F64 Implementations of Utility Traits
impl<T: Copy> Copy for DiffNum<T> {}

impl<T: Clone> Clone for DiffNum<T> {
    fn clone(&self) -> Self {
        return Self {
            f: self.f.clone(),
            df: self.df.clone(),
        };
    }
}

impl<T: Default> Default for DiffNum<T> {
    fn default() -> Self {
        return Self {
            f: T::default(),
            df: T::default(),
        };
    }
}

impl<T: std::fmt::Display> std::fmt::Display for DiffNum<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "(f: {}, df: {})", self.f, self.df);
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for DiffNum<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffNum")
            .field("f", &self.f)
            .field("df", &self.df)
            .finish()
    }
}

// Froms for DiffNum from scalar types
// From implementations for floats
impl From<f64> for DiffNum<f64> {
    fn from(value: f64) -> Self {
        Self { f: value, df: 0.0 }
    }
}

impl From<f32> for DiffNum<f32> {
    fn from(value: f32) -> Self {
        Self { f: value, df: 0.0 }
    }
}

// Mathematical Operations
impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add for DiffNum<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return Self {
            f: self.f + other.f,
            df: self.df + other.df,
        };
    }
}

impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub for DiffNum<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        return Self {
            f: self.f - other.f,
            df: self.df - other.df,
        };
    }
}

impl<T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy> std::ops::Mul for DiffNum<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        return Self {
            f: self.f * other.f,
            df: self.f * other.df + self.df * other.f,
        };
    }
}

impl<
        T: std::ops::Div<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + Copy,
    > std::ops::Div for DiffNum<T>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        return Self {
            f: self.f / other.f,
            df: (self.df * other.f - self.f * other.df) / (other.f * other.f),
        };
    }
}

impl<T: std::ops::Neg<Output = T> + Copy> std::ops::Neg for DiffNum<T> {
    type Output = Self;

    fn neg(self) -> Self {
        return Self {
            f: -self.f,
            df: -self.df,
        };
    }
}

impl<T: std::ops::AddAssign + Copy> std::ops::AddAssign for DiffNum<T> {
    fn add_assign(&mut self, other: Self) {
        self.f += other.f;
        self.df += other.df;
    }
}

impl<T: std::ops::SubAssign + Copy> std::ops::SubAssign for DiffNum<T> {
    fn sub_assign(&mut self, other: Self) {
        self.f -= other.f;
        self.df -= other.df;
    }
}

impl<
        T: std::ops::MulAssign
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + Copy,
    > std::ops::MulAssign for DiffNum<T>
{
    fn mul_assign(&mut self, other: Self) {
        let new_df: T = self.f * other.df + self.df * other.f;
        self.f *= other.f;
        self.df = new_df;
    }
}

impl<
        T: std::ops::DivAssign
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    > std::ops::DivAssign for DiffNum<T>
{
    fn div_assign(&mut self, other: Self) {
        let new_df: T = (self.df * other.f - self.f * other.df) / (other.f * other.f);
        self.f /= other.f;
        self.df = new_df;
    }
}

// Operations with scalars on the right
impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add<T> for DiffNum<T> {
    type Output = Self;

    fn add(self, other: T) -> Self {
        return Self {
            f: self.f + other,
            df: self.df,
        };
    }
}

impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub<T> for DiffNum<T> {
    type Output = Self;

    fn sub(self, other: T) -> Self {
        return Self {
            f: self.f - other,
            df: self.df,
        };
    }
}

impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul<T> for DiffNum<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        return Self {
            f: self.f * other,
            df: self.df * other,
        };
    }
}

impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div<T> for DiffNum<T> {
    type Output = Self;

    fn div(self, other: T) -> Self {
        return Self {
            f: self.f / other,
            df: self.df / other,
        };
    }
}

// Operations with scalars on the left
impl std::ops::Add<DiffNum<f64>> for f64 {
    type Output = DiffNum<f64>;

    fn add(self, other: DiffNum<f64>) -> DiffNum<f64> {
        return DiffNum {
            f: self + other.f,
            df: other.df,
        };
    }
}

impl std::ops::Sub<DiffNum<f64>> for f64 {
    type Output = DiffNum<f64>;

    fn sub(self, other: DiffNum<f64>) -> DiffNum<f64> {
        return DiffNum {
            f: self - other.f,
            df: -other.df,
        };
    }
}

impl std::ops::Mul<DiffNum<f64>> for f64 {
    type Output = DiffNum<f64>;

    fn mul(self, other: DiffNum<f64>) -> DiffNum<f64> {
        return DiffNum {
            f: self * other.f,
            df: self * other.df,
        };
    }
}

impl std::ops::Div<DiffNum<f64>> for f64 {
    type Output = DiffNum<f64>;

    fn div(self, other: DiffNum<f64>) -> DiffNum<f64> {
        return DiffNum {
            f: self / other.f,
            df: (-self * other.df) / (other.f * other.f),
        };
    }
}

// Comparison and Equality
impl<T: Eq> Eq for DiffNum<T> {}

impl<T: PartialEq> PartialEq for DiffNum<T> {
    fn eq(&self, other: &Self) -> bool {
        return self.f == other.f && self.df == other.df;
    }
}

impl<T: Ord> Ord for DiffNum<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        return self.f.cmp(&other.f);
    }
}

impl<T: PartialOrd> PartialOrd for DiffNum<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        return self.f.partial_cmp(&other.f);
    }
}

// Implementation of mathematical functions
impl DiffNum<f64> {
    pub fn powi(self, n: i32) -> Self {
        return Self {
            f: self.f.powi(n),
            df: (n as f64) * self.f.powi(n - 1) * self.df,
        };
    }

    pub fn powf(self, n: f64) -> Self {
        return Self {
            f: self.f.powf(n),
            df: n * self.f.powf(n - 1.0) * self.df,
        };
    }

    pub fn sin(self) -> Self {
        return Self {
            f: self.f.sin(),
            df: self.f.cos() * self.df,
        };
    }

    pub fn cos(self) -> Self {
        return Self {
            f: self.f.cos(),
            df: -self.f.sin() * self.df,
        };
    }

    pub fn tan(self) -> Self {
        return Self {
            f: self.f.tan(),
            df: (1.0 / self.f.cos().powi(2)) * self.df,
        };
    }

    pub fn asin(self) -> Self {
        return Self {
            f: self.f.asin(),
            df: (1.0 / (1.0 - self.f.powi(2)).sqrt()) * self.df,
        };
    }

    pub fn acos(self) -> Self {
        return Self {
            f: self.f.acos(),
            df: (-1.0 / (1.0 - self.f.powi(2)).sqrt()) * self.df,
        };
    }

    pub fn atan(self) -> Self {
        return Self {
            f: self.f.atan(),
            df: (1.0 / (1.0 + self.f.powi(2))) * self.df,
        };
    }

    pub fn exp(self) -> Self {
        return Self {
            f: self.f.exp(),
            df: self.f.exp() * self.df,
        };
    }

    pub fn ln(self) -> Self {
        return Self {
            f: self.f.ln(),
            df: (1.0 / self.f) * self.df,
        };
    }

    pub fn log10(self) -> Self {
        return Self {
            f: self.f.log10(),
            df: (1.0 / (self.f * std::f64::consts::LN_10)) * self.df,
        };
    }

    pub fn sqrt(self) -> Self {
        return Self {
            f: self.f.sqrt(),
            df: (0.5 / self.f.sqrt()) * self.df,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test Addition for f64, f32
    #[test]
    fn test_addition_f64() {
        let a: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f64> = a + b;
        assert_eq!(c.f, 5.0);
        assert_eq!(c.df, 5.0);
    }

    #[test]
    fn test_addition_f32() {
        let a: DiffNum<f32> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f32> = a + b;
        assert_eq!(c.f, 5.0);
        assert_eq!(c.df, 5.0);
    }

    #[test]
    fn test_add_assign_f64() {
        let mut a: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..3 {
            a += b;
        }
        assert_eq!(a.f, 11.0);
        assert_eq!(a.df, 13.0);
    }

    #[test]
    fn test_add_assign_f32() {
        let mut a: DiffNum<f32> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..3 {
            a += b;
        }
        assert_eq!(a.f, 11.0);
        assert_eq!(a.df, 13.0);
    }

    // Test subtraction for f64, f32
    #[test]
    fn test_subtraction_f64() {
        let a: DiffNum<f64> = DiffNum { f: 5.0, df: 4.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 1.0 };
        let c: DiffNum<f64> = a - b;
        assert_eq!(c.f, 2.0);
        assert_eq!(c.df, 3.0);
    }

    #[test]
    fn test_subtraction_f32() {
        let a: DiffNum<f32> = DiffNum { f: 5.0, df: 4.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 1.0 };
        let c: DiffNum<f32> = a - b;
        assert_eq!(c.f, 2.0);
        assert_eq!(c.df, 3.0);
    }

    #[test]
    fn test_sub_assign_f64() {
        let mut a: DiffNum<f64> = DiffNum { f: 5.0, df: 4.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 1.0 };
        for _ in 0..3 {
            a -= b;
        }
        assert_eq!(a.f, -4.0);
        assert_eq!(a.df, 1.0);
    }

    fn test_sub_assign_f32() {
        let mut a: DiffNum<f32> = DiffNum { f: 5.0, df: 4.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 1.0 };
        for _ in 0..3 {
            a -= b;
        }
        assert_eq!(a.f, -4.0);
        assert_eq!(a.df, 1.0);
    }

    // Test Multiplication for f64, f32
    #[test]
    fn test_multiplication_f64() {
        let a: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f64> = a * b;
        assert_eq!(c.f, 6.0);
        assert_eq!(c.df, 11.0);
    }

    #[test]
    fn test_multiplication_f32() {
        let a: DiffNum<f32> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f32> = a * b;
        assert_eq!(c.f, 6.0);
        assert_eq!(c.df, 11.0);
    }

    #[test]
    fn test_mul_assign_f64() {
        let mut a: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..2 {
            a *= b;
        }
        assert_eq!(a.f, 18.0);
        assert_eq!(a.df, 57.0);
    }

    #[test]
    fn test_mul_assign_f32() {
        let mut a: DiffNum<f32> = DiffNum { f: 2.0, df: 1.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..2 {
            a *= b;
        }
        assert_eq!(a.f, 18.0);
        assert_eq!(a.df, 57.0);
    }

    // Test Division for f64, f32
    #[test]
    fn test_division_f64() {
        let a: DiffNum<f64> = DiffNum { f: 6.0, df: 11.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f64> = a / b;
        assert_eq!(c.f, 2.0);
        assert_eq!(c.df, (11.0 * 3.0 - 6.0 * 4.0) / (3.0 * 3.0));
    }

    #[test]
    fn test_division_f32() {
        let a: DiffNum<f32> = DiffNum { f: 6.0, df: 11.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        let c: DiffNum<f32> = a / b;
        assert_eq!(c.f, 2.0);
        assert_eq!(c.df, (11.0 * 3.0 - 6.0 * 4.0) / (3.0 * 3.0));
    }

    #[test]
    fn test_div_assign_f64() {
        let mut a: DiffNum<f64> = DiffNum { f: 18.0, df: 49.0 };
        let b: DiffNum<f64> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..2 {
            a /= b;
        }
        assert_eq!(a.f, 2.0);
        assert!((a.df - (1.0 / 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_div_assign_f32() {
        let mut a: DiffNum<f32> = DiffNum { f: 18.0, df: 49.0 };
        let b: DiffNum<f32> = DiffNum { f: 3.0, df: 4.0 };
        for _ in 0..2 {
            a /= b;
        }
        assert_eq!(a.f, 2.0);
        assert!((a.df - (1.0 / 9.0)).abs() < 1e-6);
    }

    // Tests for Negation for f64, f32
    #[test]
    fn test_negation_f64() {
        let a: DiffNum<f64> = DiffNum { f: 2.0, df: 3.0 };
        let b: DiffNum<f64> = -a;
        assert_eq!(b.f, -2.0);
        assert_eq!(b.df, -3.0);
    }

    #[test]
    fn test_negation_f32() {
        let a: DiffNum<f32> = DiffNum { f: 2.0, df: 3.0 };
        let b: DiffNum<f32> = -a;
        assert_eq!(b.f, -2.0);
        assert_eq!(b.df, -3.0);
    }

    // Test polynomial differentiation
    #[test]
    fn test_quadratic_polynomial() {
        // f(x) = x^2 + 2x + 1
        // f'(x) = 2x + 2
        let x: DiffNum<f64> = DiffNum { f: 3.0, df: 1.0 }; // At x = 3
        let f_x: DiffNum<f64> = x.powi(2) + 2.0 * x + 1.0;
        assert_eq!(f_x.f, 16.0); // 3^2 + 2*3 + 1 = 16
        assert_eq!(f_x.df, 8.0); // f'(3) = 2*3 + 2 = 8
    }

    #[test]
    fn test_2d_quadratic_polynomial() {
        // f(x, y) = x^2 + y^2 + 2xy
        // df/dx = 2x + 2y
        // df/dy = 2y + 2x
        let mut x: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 }; // At x = 2
        let mut y: DiffNum<f64> = DiffNum { f: 3.0, df: 0.0 }; // At y = 3
        let f_dx: DiffNum<f64> = x.powi(2) + y.powi(2) + 2.0 * x * y;
        assert_eq!(f_dx.f, 25.0); // 2^2 + 3^2 + 2*2*3 = 25
        assert_eq!(f_dx.df, 10.0); // df/dx at (2,3) = 2*2 + 2*3 = 10

        x = DiffNum { f: 2.0, df: 0.0 }; // At x = 2
        y = DiffNum { f: 3.0, df: 1.0 }; // At y = 3
        let f_dy: DiffNum<f64> = x.powi(2) + y.powi(2) + 2.0 * x * y;
        assert_eq!(f_dy.f, 25.0); // 2^2 + 3^2 + 2*2*3 = 25
        assert_eq!(f_dy.df, 10.0); // df/dy at (2,3) = 2*3 + 2*2 = 10
    }

    // Tests for transcendental functions
    #[test]
    fn test_trig_funcs() {
        let x: DiffNum<f64> = DiffNum {
            f: std::f64::consts::PI / 4.0,
            df: 1.0,
        }; // 45 degrees

        let sin_x = x.sin();
        assert!((sin_x.f - (std::f64::consts::PI / 4.0).sin()).abs() < 1e-10);
        assert!((sin_x.df - (std::f64::consts::PI / 4.0).cos()).abs() < 1e-10);

        let cos_x = x.cos();
        assert!((cos_x.f - (std::f64::consts::PI / 4.0).cos()).abs() < 1e-10);
        assert!((cos_x.df + (std::f64::consts::PI / 4.0).sin()).abs() < 1e-10);

        let tan_x = x.tan();
        assert!((tan_x.f - (std::f64::consts::PI / 4.0).tan()).abs() < 1e-10);
        assert!((tan_x.df - (1.0 / (std::f64::consts::PI / 4.0).cos().powi(2))).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_and_log_funcs() {
        let x: DiffNum<f64> = DiffNum { f: 2.0, df: 1.0 };

        let exp_x = x.exp();
        assert!((exp_x.f - 2.0f64.exp()).abs() < 1e-10);
        assert!((exp_x.df - 2.0f64.exp()).abs() < 1e-10);

        let ln_x = x.ln();
        assert!((ln_x.f - 2.0f64.ln()).abs() < 1e-10);
        assert!((ln_x.df - (1.0 / 2.0)).abs() < 1e-10);

        let log10_x = x.log10();
        assert!((log10_x.f - 2.0f64.log10()).abs() < 1e-10);
        assert!((log10_x.df - (1.0 / (2.0 * std::f64::consts::LN_10))).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_func() {
        let x: DiffNum<f64> = DiffNum { f: 4.0, df: 1.0 };

        let sqrt_x = x.sqrt();
        assert!((sqrt_x.f - 2.0).abs() < 1e-10);
        assert!((sqrt_x.df - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_powi_func() {
        let x: DiffNum<f64> = DiffNum { f: 3.0, df: 1.0 };

        let powi_x = x.powi(3);
        assert!((powi_x.f - 27.0).abs() < 1e-10);
        assert!((powi_x.df - 27.0).abs() < 1e-10); // 3 * 3^2 * 1 = 27
    }

    #[test]
    fn test_powf_func() {
        let x: DiffNum<f64> = DiffNum { f: 4.0, df: 1.0 };

        let powf_x = x.powf(2.5);
        assert!((powf_x.f - 32.0).abs() < 1e-10);
        assert!((powf_x.df - 20.0).abs() < 1e-10); // 2.5 * 4^1.5 * 1 = 20
    }

    #[test]
    fn test_arc_trig_funcs() {
        let x: DiffNum<f64> = DiffNum { f: 0.5, df: 1.0 };

        let asin_x = x.asin();
        assert!((asin_x.f - 0.5f64.asin()).abs() < 1e-10);
        assert!((asin_x.df - (1.0 / (1.0 - 0.5f64.powi(2)).sqrt())).abs() < 1e-10);

        let acos_x = x.acos();
        assert!((acos_x.f - 0.5f64.acos()).abs() < 1e-10);
        assert!((acos_x.df + (1.0 / (1.0 - 0.5f64.powi(2)).sqrt())).abs() < 1e-10);

        let atan_x = x.atan();
        assert!((atan_x.f - 0.5f64.atan()).abs() < 1e-10);
        assert!((atan_x.df - (1.0 / (1.0 + 0.5f64.powi(2)))).abs() < 1e-10);
    }

    #[test]
    fn test_chain_rule() {
        // f(x) = sin(x^2)
        // f'(x) = cos(x^2) * 2x
        let x: DiffNum<f64> = DiffNum {
            f: std::f64::consts::PI / 4.0,
            df: 1.0,
        };
        let f_x: DiffNum<f64> = (x.powi(2)).sin();
        assert!((f_x.f - (std::f64::consts::PI / 4.0).powi(2).sin()).abs() < 1e-10);
        assert!(
            (f_x.df
                - (std::f64::consts::PI / 4.0).powi(2).cos() * 2.0 * (std::f64::consts::PI / 4.0))
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_chain_rule_multiple_vars() {
        // f(x, y) = exp(x * y)
        // df/dx = y * exp(x * y)
        // df/dy = x * exp(x * y)
        let mut x: DiffNum<f64> = DiffNum { f: 1.0, df: 1.0 }; // At x = 1
        let mut y: DiffNum<f64> = DiffNum { f: 2.0, df: 0.0 }; // At y = 2
        let f_dx: DiffNum<f64> = (x * y).exp();
        assert!((f_dx.f - (1.0f64 * 2.0f64).exp()).abs() < 1e-10);
        assert!((f_dx.df - (2.0f64 * (1.0f64 * 2.0f64).exp())).abs() < 1e-10);

        x = DiffNum { f: 1.0, df: 0.0 }; // At x = 1
        y = DiffNum { f: 2.0, df: 1.0 }; // At y = 2
        let f_dy: DiffNum<f64> = (x * y).exp();
        assert!((f_dy.f - (1.0f64 * 2.0f64).exp()).abs() < 1e-10);
        assert!((f_dy.df - (1.0f64 * (1.0f64 * 2.0f64).exp())).abs() < 1e-10);
    }
}
