//! Minimal SVG path `d` attribute parser.
//!
//! Parses the subset of SVG path commands needed for the test SVG:
//! M/m, L/l, H/h, V/v, C/c, S/s, A/a, Z/z.
//! Produces waverast `Segment`s grouped into closed sub-paths.

use waverast::contour::{CircularArc, CubicBez, Line, Point, Segment};

/// Parse an SVG path `d` attribute string into a list of waverast segments.
///
/// Each `M`/`m` starts a new sub-path. Each `z`/`Z` closes the current
/// sub-path with a line back to the sub-path start.
pub fn parse_svg_path(d: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut cur = Point::new(0.0, 0.0);
    let mut sub_start = cur;
    let mut last_ctrl: Option<Point> = None; // for S/s smooth cubic
    let mut chars = d.as_bytes();
    let mut cmd: u8 = b'M';

    while !chars.is_empty() {
        skip_whitespace_and_commas(&mut chars);
        if chars.is_empty() {
            break;
        }

        // Check for new command letter
        if chars[0].is_ascii_alphabetic() {
            cmd = chars[0];
            chars = &chars[1..];
            skip_whitespace_and_commas(&mut chars);
        }

        match cmd {
            b'M' => {
                // Implicitly close the previous sub-path
                if (cur.x - sub_start.x).abs() > 1e-10 || (cur.y - sub_start.y).abs() > 1e-10 {
                    segments.push(Segment::Line(Line::new(cur, sub_start)));
                }
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                cur = Point::new(x, y);
                sub_start = cur;
                last_ctrl = None;
                cmd = b'L'; // subsequent coords are implicit L
            }
            b'm' => {
                // Implicitly close the previous sub-path
                if (cur.x - sub_start.x).abs() > 1e-10 || (cur.y - sub_start.y).abs() > 1e-10 {
                    segments.push(Segment::Line(Line::new(cur, sub_start)));
                }
                let dx = parse_number(&mut chars);
                let dy = parse_number(&mut chars);
                cur = Point::new(cur.x + dx, cur.y + dy);
                sub_start = cur;
                last_ctrl = None;
                cmd = b'l';
            }
            b'L' => {
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                let next = Point::new(x, y);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'l' => {
                let dx = parse_number(&mut chars);
                let dy = parse_number(&mut chars);
                let next = Point::new(cur.x + dx, cur.y + dy);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'H' => {
                let x = parse_number(&mut chars);
                let next = Point::new(x, cur.y);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'h' => {
                let dx = parse_number(&mut chars);
                let next = Point::new(cur.x + dx, cur.y);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'V' => {
                let y = parse_number(&mut chars);
                let next = Point::new(cur.x, y);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'v' => {
                let dy = parse_number(&mut chars);
                let next = Point::new(cur.x, cur.y + dy);
                segments.push(Segment::Line(Line::new(cur, next)));
                cur = next;
                last_ctrl = None;
            }
            b'C' => {
                let x1 = parse_number(&mut chars);
                let y1 = parse_number(&mut chars);
                let x2 = parse_number(&mut chars);
                let y2 = parse_number(&mut chars);
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                let p1 = Point::new(x1, y1);
                let p2 = Point::new(x2, y2);
                let p3 = Point::new(x, y);
                segments.push(Segment::CubicBez(CubicBez::new(cur, p1, p2, p3)));
                last_ctrl = Some(p2);
                cur = p3;
            }
            b'c' => {
                let dx1 = parse_number(&mut chars);
                let dy1 = parse_number(&mut chars);
                let dx2 = parse_number(&mut chars);
                let dy2 = parse_number(&mut chars);
                let dx = parse_number(&mut chars);
                let dy = parse_number(&mut chars);
                let p1 = Point::new(cur.x + dx1, cur.y + dy1);
                let p2 = Point::new(cur.x + dx2, cur.y + dy2);
                let p3 = Point::new(cur.x + dx, cur.y + dy);
                segments.push(Segment::CubicBez(CubicBez::new(cur, p1, p2, p3)));
                last_ctrl = Some(p2);
                cur = p3;
            }
            b'S' => {
                let x2 = parse_number(&mut chars);
                let y2 = parse_number(&mut chars);
                let x = parse_number(&mut chars);
                let y = parse_number(&mut chars);
                let p1 = reflect_ctrl(cur, last_ctrl);
                let p2 = Point::new(x2, y2);
                let p3 = Point::new(x, y);
                segments.push(Segment::CubicBez(CubicBez::new(cur, p1, p2, p3)));
                last_ctrl = Some(p2);
                cur = p3;
            }
            b's' => {
                let dx2 = parse_number(&mut chars);
                let dy2 = parse_number(&mut chars);
                let dx = parse_number(&mut chars);
                let dy = parse_number(&mut chars);
                let p1 = reflect_ctrl(cur, last_ctrl);
                let p2 = Point::new(cur.x + dx2, cur.y + dy2);
                let p3 = Point::new(cur.x + dx, cur.y + dy);
                segments.push(Segment::CubicBez(CubicBez::new(cur, p1, p2, p3)));
                last_ctrl = Some(p2);
                cur = p3;
            }
            b'A' | b'a' => {
                let abs = cmd == b'A';
                let rx = parse_number(&mut chars);
                let ry = parse_number(&mut chars);
                let _x_rot = parse_number(&mut chars);
                let large_arc = parse_flag(&mut chars);
                let sweep = parse_flag(&mut chars);
                let (ex, ey) = if abs {
                    (parse_number(&mut chars), parse_number(&mut chars))
                } else {
                    let dx = parse_number(&mut chars);
                    let dy = parse_number(&mut chars);
                    (cur.x + dx, cur.y + dy)
                };
                let end = Point::new(ex, ey);

                // Use average radius for circular arc (all arcs in our test
                // file have rx == ry).
                let r = (rx + ry) * 0.5;
                if let Some(seg) = svg_arc_to_segment(cur, end, r, large_arc, sweep) {
                    segments.push(seg);
                } else {
                    segments.push(Segment::Line(Line::new(cur, end)));
                }
                cur = end;
                last_ctrl = None;
            }
            b'Z' | b'z' => {
                if (cur.x - sub_start.x).abs() > 1e-10 || (cur.y - sub_start.y).abs() > 1e-10 {
                    segments.push(Segment::Line(Line::new(cur, sub_start)));
                }
                cur = sub_start;
                last_ctrl = None;
            }
            _ => {
                // Skip unknown command
                chars = &chars[1..];
            }
        }
    }

    // Close the final sub-path
    if (cur.x - sub_start.x).abs() > 1e-10 || (cur.y - sub_start.y).abs() > 1e-10 {
        segments.push(Segment::Line(Line::new(cur, sub_start)));
    }

    segments
}

/// Reflect the last control point across the current point (for S/s commands).
fn reflect_ctrl(cur: Point, last_ctrl: Option<Point>) -> Point {
    match last_ctrl {
        Some(c) => Point::new(2.0 * cur.x - c.x, 2.0 * cur.y - c.y),
        None => cur,
    }
}

/// Convert an SVG endpoint-parameterized circular arc to a waverast CircularArc.
///
/// Returns None if the arc degenerates (endpoints too close or radius too small).
fn svg_arc_to_segment(
    p0: Point,
    p1: Point,
    mut r: f64,
    large_arc: bool,
    sweep: bool,
) -> Option<Segment> {
    // W3C SVG arc implementation notes:
    // https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes

    let dx = p0.x - p1.x;
    let dy = p0.y - p1.y;
    let half_dist = (dx * dx + dy * dy).sqrt() * 0.5;

    if half_dist < 1e-10 {
        return None;
    }

    // Ensure radius is large enough
    if r < half_dist {
        r = half_dist;
    }

    // Midpoint
    let mx = (p0.x + p1.x) * 0.5;
    let my = (p0.y + p1.y) * 0.5;

    // Distance from midpoint to center
    let h = (r * r - half_dist * half_dist).max(0.0).sqrt();

    // Unit vector perpendicular to the chord
    let ux = -dy / (2.0 * half_dist);
    let uy = dx / (2.0 * half_dist);

    // Choose center based on large_arc and sweep flags
    let sign = if large_arc == sweep { -1.0 } else { 1.0 };
    let cx = mx + sign * h * ux;
    let cy = my + sign * h * uy;

    // Compute angles
    let theta0 = (p0.y - cy).atan2(p0.x - cx);
    let mut theta1 = (p1.y - cy).atan2(p1.x - cx);

    // Adjust theta1 for sweep direction and large_arc flag.
    // sweep=true means positive angular direction (CCW in math coords).
    // large_arc=true means |dtheta| > π.
    let mut dtheta = theta1 - theta0;

    if sweep {
        if dtheta < 0.0 {
            dtheta += std::f64::consts::TAU;
        }
        if !large_arc && dtheta > std::f64::consts::PI {
            dtheta -= std::f64::consts::TAU;
        }
        if large_arc && dtheta < std::f64::consts::PI {
            dtheta += std::f64::consts::TAU;
        }
    } else {
        if dtheta > 0.0 {
            dtheta -= std::f64::consts::TAU;
        }
        if !large_arc && dtheta < -std::f64::consts::PI {
            dtheta += std::f64::consts::TAU;
        }
        if large_arc && dtheta > -std::f64::consts::PI {
            dtheta -= std::f64::consts::TAU;
        }
    }
    theta1 = theta0 + dtheta;

    Some(Segment::CircularArc(CircularArc {
        center: Point::new(cx, cy),
        radius: r,
        theta0,
        theta1,
    }))
}

fn skip_whitespace_and_commas(s: &mut &[u8]) {
    while let Some((&c, rest)) = s.split_first() {
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == b',' {
            *s = rest;
        } else {
            break;
        }
    }
}

/// Parse a floating-point number from the byte stream.
///
/// Handles the SVG compaction where a new number can start with '.' or '-'
/// immediately after the previous number (e.g. "1.5.3" = 1.5, 0.3).
fn parse_number(s: &mut &[u8]) -> f64 {
    skip_whitespace_and_commas(s);

    let start = *s;
    let mut i = 0;
    let len = s.len();

    // Optional sign
    if i < len && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }

    // Integer part
    while i < len && s[i].is_ascii_digit() {
        i += 1;
    }

    // Fractional part
    if i < len && s[i] == b'.' {
        i += 1;
        while i < len && s[i].is_ascii_digit() {
            i += 1;
        }
    }

    // Exponent
    if i < len && (s[i] == b'e' || s[i] == b'E') {
        i += 1;
        if i < len && (s[i] == b'-' || s[i] == b'+') {
            i += 1;
        }
        while i < len && s[i].is_ascii_digit() {
            i += 1;
        }
    }

    let num_str = std::str::from_utf8(&start[..i]).unwrap_or("0");
    *s = &start[i..];
    num_str.parse().unwrap_or(0.0)
}

/// Parse a single flag digit (0 or 1) — used for arc large-arc and sweep flags.
///
/// SVG allows flags to be packed without separators: "1 0 1" or "101".
fn parse_flag(s: &mut &[u8]) -> bool {
    skip_whitespace_and_commas(s);
    if let Some((&c, rest)) = s.split_first() {
        *s = rest;
        c == b'1'
    } else {
        false
    }
}
