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
// From implementations for all primitive numeric types
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

impl From<i8> for DiffNum<i8> {
    fn from(value: i8) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<i16> for DiffNum<i16> {
    fn from(value: i16) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<i32> for DiffNum<i32> {
    fn from(value: i32) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<i64> for DiffNum<i64> {
    fn from(value: i64) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<i128> for DiffNum<i128> {
    fn from(value: i128) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<isize> for DiffNum<isize> {
    fn from(value: isize) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<u8> for DiffNum<u8> {
    fn from(value: u8) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<u16> for DiffNum<u16> {
    fn from(value: u16) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<u32> for DiffNum<u32> {
    fn from(value: u32) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<u64> for DiffNum<u64> {
    fn from(value: u64) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<u128> for DiffNum<u128> {
    fn from(value: u128) -> Self {
        Self { f: value, df: 0 }
    }
}

impl From<usize> for DiffNum<usize> {
    fn from(value: usize) -> Self {
        Self { f: value, df: 0 }
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
// impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add<DiffNum<T>> for T {
//     type Output = DiffNum<T>;

//     fn add(self, other: DiffNum<T>) -> DiffNum<T> {
//         return DiffNum {
//             f: self + other.f,
//             df: other.df,
//         };
//     }
// }

// impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub<DiffNum<T>> for T {
//     type Output = DiffNum<T>;

//     fn sub(self, other: DiffNum<T>) -> DiffNum<T> {
//         return DiffNum {
//             f: self - other.f,
//             df: -other.df,
//         };
//     }
// }

// impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul<DiffNum<T>> for T {
//     type Output = DiffNum<T>;

//     fn mul(self, other: DiffNum<T>) -> DiffNum<T> {
//         return DiffNum {
//             f: self * other.f,
//             df: self * other.df,
//         };
//     }
// }

// impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div<DiffNum<T>> for T {
//     type Output = DiffNum<T>;

//     fn div(self, other: DiffNum<T>) -> DiffNum<T> {
//         return DiffNum {
//             f: self / other.f,
//             df: (-self * other.df) / (other.f * other.f),
//         };
//     }
// }

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

    // Test polynomial differentiation
    #[test]
    fn test_quadratic_polynomial() {
        // f(x) = x^2 + 2x + 1
        // f'(x) = 2x + 2
        let x: DiffNum<f64> = DiffNum { f: 3.0, df: 1.0 }; // At x = 3
        let f_x: DiffNum<f64> = x.powi(2) + x * 2.0 + 1.0;
        assert_eq!(f_x.f, 16.0); // 3^2 + 2*3 + 1 = 16
        assert_eq!(f_x.df, 8.0); // f'(3) = 2*3 + 2 = 8
    }
}
