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
