//! math4d.rs — 4D affine transforms as (Mat4 A, Vec4 t): p' = A p + t.
//!
//! No 5×5 homogeneous matrices: a 4D affine map is a 4×4 linear part plus a translation.
//! Rotations in 4D happen in *planes*, not around axes: there are C(4,2)=6 of them.
//! XY/XZ/YZ are the familiar 3D rotations; XW/YW/ZW rotate "into the 4th dimension" and
//! make cross-sections morph as the object turns through our w = w_slice hyperplane.
use glam::{Mat4, Vec4};

/// The six coordinate rotation planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane { XY, XZ, XW, YZ, YW, ZW }

impl Plane {
    /// Parse "xy", "xw", ... (used by Lua).
    pub fn parse(s: &str) -> Option<Plane> {
        Some(match s.to_ascii_lowercase().as_str() {
            "xy" => Plane::XY, "xz" => Plane::XZ, "xw" => Plane::XW,
            "yz" => Plane::YZ, "yw" => Plane::YW, "zw" => Plane::ZW,
            _ => return None,
        })
    }
    /// Axis indices (i, j) of the plane; rotation mixes components i and j only.
    pub fn axes(self) -> (usize, usize) {
        match self {
            Plane::XY => (0, 1), Plane::XZ => (0, 2), Plane::XW => (0, 3),
            Plane::YZ => (1, 2), Plane::YW => (1, 3), Plane::ZW => (2, 3),
        }
    }
}

/// p' = a * p + t
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine4 {
    pub a: Mat4,
    pub t: Vec4,
}

impl Default for Affine4 {
    fn default() -> Self { Self::IDENTITY }
}

impl Affine4 {
    pub const IDENTITY: Affine4 = Affine4 { a: Mat4::IDENTITY, t: Vec4::ZERO };

    /// Rotation by `angle` radians in `plane`: identity except a 2×2 block
    /// [cos −sin; sin cos] at rows/cols (i, j).
    pub fn rotation(plane: Plane, angle: f32) -> Affine4 {
        let (i, j) = plane.axes();
        let (s, c) = angle.sin_cos();
        let mut m = Mat4::IDENTITY.to_cols_array_2d(); // m[col][row]
        m[i][i] = c; m[j][j] = c;
        m[i][j] = s;  // column i, row j
        m[j][i] = -s; // column j, row i
        Affine4 { a: Mat4::from_cols_array_2d(&m), t: Vec4::ZERO }
    }

    /// Shear: component `dst` += k * component `src` (e.g. dst=w, src=x tilts the object through w).
    pub fn shear(dst: usize, src: usize, k: f32) -> Affine4 {
        let mut m = Mat4::IDENTITY.to_cols_array_2d();
        m[src][dst] += k;
        Affine4 { a: Mat4::from_cols_array_2d(&m), t: Vec4::ZERO }
    }

    pub fn translation(t: Vec4) -> Affine4 { Affine4 { a: Mat4::IDENTITY, t } }

    pub fn scale(s: Vec4) -> Affine4 { Affine4 { a: Mat4::from_diagonal(s), t: Vec4::ZERO } }

    /// `self ∘ first` — apply `first`, then `self`: (A2,t2)∘(A1,t1) = (A2A1, A2t1 + t2).
    pub fn compose(&self, first: &Affine4) -> Affine4 {
        Affine4 { a: self.a * first.a, t: self.a * first.t + self.t }
    }

    /// Inverse map: p = A⁻¹(p' − t)  ⇒  (A⁻¹, −A⁻¹t).
    pub fn inverse(&self) -> Affine4 {
        let ai = self.a.inverse();
        Affine4 { a: ai, t: -(ai * self.t) }
    }

    pub fn apply(&self, p: Vec4) -> Vec4 { self.a * p + self.t }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_roundtrip() {
        let m = Affine4::rotation(Plane::XW, 0.7)
            .compose(&Affine4::shear(3, 0, 0.3))
            .compose(&Affine4::translation(Vec4::new(1.0, 2.0, 3.0, 4.0)));
        let p = Vec4::new(0.3, -1.0, 2.0, 0.5);
        assert!((m.inverse().apply(m.apply(p)) - p).length() < 1e-5);
        assert!((m.compose(&m.inverse()).apply(p) - p).length() < 1e-5);
    }

    #[test]
    fn rotations_are_orthogonal() {
        for plane in [Plane::XY, Plane::XZ, Plane::XW, Plane::YZ, Plane::YW, Plane::ZW] {
            let r = Affine4::rotation(plane, 1.1).a;
            assert!((r.transpose() * r).abs_diff_eq(Mat4::IDENTITY, 1e-5), "{plane:?}");
            assert!((r.determinant() - 1.0).abs() < 1e-5);
            // Only the plane's two axes move; the other two are fixed.
            let (i, j) = plane.axes();
            for k in (0..4).filter(|&k| k != i && k != j) {
                let mut e = Vec4::ZERO; e[k] = 1.0;
                assert!((r * e - e).length() < 1e-6);
            }
        }
        // XW by 90° sends x to w.
        let r = Affine4::rotation(Plane::XW, std::f32::consts::FRAC_PI_2);
        assert!((r.apply(Vec4::X) - Vec4::W).length() < 1e-6);
    }

    #[test]
    fn compose_order_and_primitives() {
        let t = Affine4::translation(Vec4::new(1.0, 0.0, 0.0, 0.0));
        let s = Affine4::scale(Vec4::splat(2.0));
        // s∘t: translate first, then scale: (0 + 1) * 2 = 2
        assert!((s.compose(&t).apply(Vec4::ZERO) - Vec4::new(2.0, 0.0, 0.0, 0.0)).length() < 1e-6);
        // shear w += 0.5 x
        let sh = Affine4::shear(3, 0, 0.5);
        assert!((sh.apply(Vec4::new(2.0, 0.0, 0.0, 0.0)) - Vec4::new(2.0, 0.0, 0.0, 1.0)).length() < 1e-6);
    }
}
