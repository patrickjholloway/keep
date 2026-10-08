//! field.rs — 4D implicit fields F(q) (signed-distance-like), CPU mirror of field.comp.
//!
//! A `Field` is a flat, ordered list of primitives folded left-to-right with a CSG op —
//! exactly the shape the GPU wants (`Primitive prims[MAX_PRIMS]`), so no tree flattening.
//! CPU `eval` exists for tests and for sanity-checking the shader math.
use glam::{Vec2, Vec4, Vec4Swizzles};

use crate::{math4d::Affine4, scene::{GpuPrimitive, MAX_PRIMS}};

/// 4D primitive shapes. Kind ids must match PRIM_* in shaders/common.glsl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// |q| = r. Its 3D slices are spheres that grow then shrink as w passes through.
    Hypersphere { r: f32 },
    /// 4D box with half extents; slices are cubes/prisms, rotated slices get faceted.
    Tesseract { half: Vec4 },
    /// Product of two discs (xy radius r1, zw radius r2) — asymmetric between planes.
    Duocylinder { r1: f32, r2: f32 },
    /// Thickened Clifford torus |q.xy| = r1, |q.zw| = r2 — a flat torus living in 4D.
    Clifford { r1: f32, r2: f32, thickness: f32 },
}

/// How a primitive combines with everything before it. Ids match OP_* in common.glsl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Union,
    /// Polynomial smooth-min with blend radius k.
    SmoothUnion { k: f32 },
    Intersect,
    Subtract,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prim {
    pub shape: Shape,
    pub op: Op,
    /// Local transform (forward); packed as its inverse.
    pub local: Affine4,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Field {
    pub prims: Vec<Prim>,
}

impl Shape {
    pub fn kind_id(&self) -> u32 {
        match self { Shape::Hypersphere { .. } => 0, Shape::Tesseract { .. } => 1, Shape::Duocylinder { .. } => 2, Shape::Clifford { .. } => 3 }
    }
    fn params(&self) -> [f32; 4] {
        match *self {
            Shape::Hypersphere { r } => [r, 0.0, 0.0, 0.0],
            Shape::Tesseract { half } => half.to_array(),
            Shape::Duocylinder { r1, r2 } => [r1, r2, 0.0, 0.0],
            Shape::Clifford { r1, r2, thickness } => [r1, r2, thickness, 0.0],
        }
    }
    /// Signed distance in primitive-local coordinates.
    pub fn sdf(&self, q: Vec4) -> f32 {
        match *self {
            Shape::Hypersphere { r } => q.length() - r,
            Shape::Tesseract { half } => {
                let d = q.abs() - half;
                d.max(Vec4::ZERO).length() + d.max_element().min(0.0)
            }
            Shape::Duocylinder { r1, r2 } => {
                let d = Vec2::new(q.xy().length() - r1, q.zw().length() - r2);
                d.max(Vec2::ZERO).length() + d.max_element().min(0.0)
            }
            Shape::Clifford { r1, r2, thickness } => {
                Vec2::new(q.xy().length() - r1, q.zw().length() - r2).length() - thickness
            }
        }
    }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k.max(1e-5)).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

impl Field {
    pub fn single(shape: Shape) -> Field {
        Field { prims: vec![Prim { shape, op: Op::Union, local: Affine4::IDENTITY }] }
    }

    /// Evaluate F at `q` (already in object space, i.e. after the global inverse transform).
    pub fn eval(&self, q: Vec4) -> f32 {
        let mut d = f32::MAX;
        for (i, p) in self.prims.iter().enumerate() {
            let di = p.shape.sdf(p.local.inverse().apply(q));
            d = if i == 0 { di } else {
                match p.op {
                    Op::Union => d.min(di),
                    Op::SmoothUnion { k } => smin(d, di, k),
                    Op::Intersect => d.max(di),
                    Op::Subtract => d.max(-di),
                }
            };
        }
        d
    }

    /// Central-difference 4D gradient with step `h` (the shader uses h = 1e-3 on xyz only).
    /// Its xyz part, normalized, is the visible surface normal of the 3D slice.
    pub fn gradient(&self, q: Vec4, h: f32) -> Vec4 {
        let e = |i: usize| { let mut v = Vec4::ZERO; v[i] = h; v };
        Vec4::from_array(std::array::from_fn(|i| (self.eval(q + e(i)) - self.eval(q - e(i))) / (2.0 * h)))
    }

    /// Evaluate at a world point after the global transform `global` (forward), exactly as
    /// field.comp does: q = A^-1 (p4 - t).
    pub fn eval_world(&self, global: &Affine4, p4: Vec4) -> f32 {
        self.eval(global.inverse().apply(p4))
    }

    /// Write into the GPU array; returns the primitive count (extra prims beyond MAX_PRIMS are dropped).
    pub fn pack(&self, out: &mut [GpuPrimitive; MAX_PRIMS]) -> usize {
        let n = self.prims.len().min(MAX_PRIMS);
        for (dst, p) in out.iter_mut().zip(&self.prims) {
            let (op, k) = match p.op { Op::Union => (0, 0.0), Op::SmoothUnion { k } => (1, k), Op::Intersect => (2, 0.0), Op::Subtract => (3, 0.0) };
            let mut params = p.shape.params();
            if op == 1 { params[3] = k; }
            *dst = GpuPrimitive {
                kind_op: [p.shape.kind_id(), op, 0, 0],
                params,
                inv_a: p.local.a.inverse().to_cols_array_2d(),
                t: p.local.t.to_array(),
            };
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math4d::Plane;

    fn close(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }

    #[test]
    fn known_values() {
        let s = Shape::Hypersphere { r: 1.0 };
        assert!(close(s.sdf(Vec4::ZERO), -1.0));
        assert!(close(s.sdf(Vec4::new(0.0, 0.0, 0.0, 2.0)), 1.0));
        let t = Shape::Tesseract { half: Vec4::splat(1.0) };
        assert!(close(t.sdf(Vec4::ZERO), -1.0));
        assert!(close(t.sdf(Vec4::new(2.0, 0.0, 0.0, 0.0)), 1.0));
        assert!(close(t.sdf(Vec4::new(2.0, 2.0, 1.0, 1.0)), 2f32.sqrt())); // edge-corner distance
        let d = Shape::Duocylinder { r1: 1.0, r2: 0.5 };
        assert!(close(d.sdf(Vec4::ZERO), -0.5));
        assert!(close(d.sdf(Vec4::new(1.0, 0.0, 0.5, 0.0)), 0.0));
        let c = Shape::Clifford { r1: 1.0, r2: 1.0, thickness: 0.1 };
        // A point on the flat torus |xy| = |zw| = 1 sits mid-tube.
        let on = Vec4::new(0.6, 0.8, 0.0, 1.0);
        assert!(close(c.sdf(on), -0.1));
    }

    #[test]
    fn csg_ops() {
        let a = Prim { shape: Shape::Hypersphere { r: 1.0 }, op: Op::Union, local: Affine4::IDENTITY };
        let b = Prim { local: Affine4::translation(Vec4::new(1.5, 0.0, 0.0, 0.0)), ..a };
        let p = Vec4::new(0.75, 0.0, 0.0, 0.0);
        let f = |op| Field { prims: vec![a, Prim { op, ..b }] }.eval(p);
        assert!(close(f(Op::Union), -0.25));
        assert!(close(f(Op::Intersect), -0.25));
        assert!(close(f(Op::Subtract), 0.25));
        assert!(f(Op::SmoothUnion { k: 0.5 }) < f(Op::Union)); // smooth union bulges outward
    }

    #[test]
    fn gradient_of_sphere_is_radial() {
        let f = Field::single(Shape::Hypersphere { r: 1.0 });
        let p = Vec4::new(0.3, -0.4, 0.5, 0.7);
        let g = f.gradient(p, 1e-3);
        assert!((g.normalize() - p.normalize()).length() < 1e-3);
        assert!((g.length() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn local_transform_matches_pack_convention() {
        // GPU computes l = inv_a * (q - t) with forward t; CPU uses local.inverse().apply(q).
        let local = Affine4::rotation(Plane::XW, 0.4).compose(&Affine4::scale(Vec4::new(1.0, 2.0, 1.0, 0.5)));
        let local = Affine4 { t: Vec4::new(0.2, 0.1, -0.3, 0.4), ..local };
        let f = Field { prims: vec![Prim { shape: Shape::Tesseract { half: Vec4::splat(0.5) }, op: Op::Union, local }] };
        let mut out = [<GpuPrimitive as bytemuck::Zeroable>::zeroed(); MAX_PRIMS];
        assert_eq!(f.pack(&mut out), 1);
        let inv_a = glam::Mat4::from_cols_array_2d(&out[0].inv_a);
        let q = Vec4::new(0.3, 0.2, -0.1, 0.6);
        let gpu_l = inv_a * (q - Vec4::from_array(out[0].t));
        assert!((gpu_l - local.inverse().apply(q)).length() < 1e-5);
    }

    /// Count lattice points within `eps` of the surface on the slice w = w0.
    fn slice_count(f: &Field, g: &Affine4, w0: f32, eps: f32) -> usize {
        let n = 40;
        let mut c = 0;
        for i in 0..n { for j in 0..n { for k in 0..n {
            let s = |v: usize| -1.5 + 3.0 * v as f32 / (n - 1) as f32;
            if f.eval_world(g, Vec4::new(s(i), s(j), s(k), w0)).abs() < eps { c += 1; }
        }}}
        c
    }

    #[test]
    fn xw_rotation_changes_slice() {
        // An axis-aligned tesseract sliced at w=0 is a cube; rotating it 45° in XW
        // stretches the cross-section along x, so the set of near-surface samples changes.
        let f = Field::single(Shape::Tesseract { half: Vec4::new(0.5, 0.5, 0.5, 0.5) });
        let a = slice_count(&f, &Affine4::IDENTITY, 0.0, 0.05);
        let b = slice_count(&f, &Affine4::rotation(Plane::XW, std::f32::consts::FRAC_PI_4), 0.0, 0.05);
        assert!(a > 0 && b > 0);
        assert!((a as i64 - b as i64).abs() > a as i64 / 10, "a={a} b={b}");
        // and a w-slice beyond the shape is empty
        assert_eq!(slice_count(&f, &Affine4::IDENTITY, 1.0, 0.05), 0);
    }
}
