#![cfg_attr(feature = "gpu", feature(f16))]

//! Wavelet rasterization of 2D shapes.
//!
//! Implements the analytic wavelet rasterization method (Manson & Schaefer, 2011),
//! which computes exact Haar wavelet coefficients of a shape's indicator function
//! via boundary line integrals (divergence theorem), then synthesizes per-pixel
//! coverage by inverse wavelet transform.
//!
//! Supports line segments, quadratic and cubic Bezier curves, circular arcs,
//! and convex superellipses as boundary primitives.

#[allow(clippy::too_many_arguments)]
pub mod contour;
pub mod rasterizer;
#[allow(clippy::too_many_arguments)]
pub mod reference;
pub mod solver;

#[cfg(feature = "gpu")]
pub mod gpu;
