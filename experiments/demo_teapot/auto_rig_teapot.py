import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh


@dataclass
class Circle3D:
    center: np.ndarray  # (3,)
    radius: float


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def rotation_around_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def fit_circle_xz(points: np.ndarray) -> Circle3D:
    """Fit a circle in XZ by simple center+median radius; Y from point mean."""
    cx = float(np.mean(points[:, 0]))
    cz = float(np.mean(points[:, 2]))
    cy = float(np.mean(points[:, 1]))
    rr = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 2] - cz) ** 2)
    r = float(np.median(rr))
    return Circle3D(center=np.array([cx, cy, cz], dtype=np.float64), radius=r)


def split_inner_outer_by_gap(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split radial samples into inner/outer groups by largest gap in sorted r."""
    idx = np.argsort(r)
    rs = r[idx]
    if len(rs) < 16:
        mid = len(rs) // 2
        return idx[:mid], idx[mid:]

    lo = int(0.10 * len(rs))
    hi = int(0.90 * len(rs))
    if hi - lo < 4:
        mid = len(rs) // 2
        return idx[:mid], idx[mid:]

    dr = np.diff(rs[lo:hi])
    k = int(np.argmax(dr)) + lo
    return idx[: k + 1], idx[k + 1 :]


def detect_body_top_inner_ring(body_mesh: trimesh.Trimesh, top_quantile: float = 0.94) -> Circle3D:
    """Detect top inner ring for watertight body from top band.

    Procedure:
    1) sample top band;
    2) project to XZ and split radial distribution into inner/outer;
    3) use inner group as ring.
    """
    v = body_mesh.vertices
    y_thr = np.quantile(v[:, 1], top_quantile)
    top_pts = v[v[:, 1] >= y_thr]
    if len(top_pts) < 64:
        y_thr = np.quantile(v[:, 1], 0.90)
        top_pts = v[v[:, 1] >= y_thr]
    if len(top_pts) < 32:
        raise RuntimeError("Not enough top-band points to detect body inner ring.")

    c0 = np.mean(top_pts[:, [0, 2]], axis=0)
    r = np.sqrt((top_pts[:, 0] - c0[0]) ** 2 + (top_pts[:, 2] - c0[1]) ** 2)
    inner_idx, _ = split_inner_outer_by_gap(r)
    inner_pts = top_pts[inner_idx]
    if len(inner_pts) < 16:
        inner_pts = top_pts

    circle = fit_circle_xz(inner_pts)
    return circle


def preprocess_lid_to_local_base(lid_mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Circle3D, float]:
    """Build lid local frame per user spec.

    - Diameter from X/Z extents;
    - Move lid base circle y to 0;
    - Define O=(0,0,0) as lid base point.
    """
    lid = lid_mesh.copy()
    ext = lid.extents
    d_x = float(ext[0])
    d_z = float(ext[2])
    d = 0.5 * (d_x + d_z)
    r = 0.5 * d

    v = lid.vertices
    x_center = float((v[:, 0].min() + v[:, 0].max()) * 0.5)
    z_center = float((v[:, 2].min() + v[:, 2].max()) * 0.5)

    # Base y from lower band to resist outliers.
    y_base = float(np.quantile(v[:, 1], 0.08))

    t = np.array([x_center, y_base, z_center], dtype=np.float64)
    lid.vertices = v - t

    base_circle = Circle3D(center=np.array([0.0, 0.0, 0.0], dtype=np.float64), radius=r)
    return lid, base_circle, r


def align_lid_to_body_inner_ring(
    lid_local: trimesh.Trimesh,
    lid_r: float,
    body_inner: Circle3D,
) -> Tuple[trimesh.Trimesh, float]:
    if lid_r <= 1e-8:
        raise RuntimeError("Invalid lid radius detected.")

    scale = body_inner.radius / lid_r
    scale = float(np.clip(scale, 0.6, 1.8))

    lid = lid_local.copy()
    lid.vertices = lid.vertices * scale
    lid.vertices = lid.vertices + body_inner.center
    return lid, scale


def hinge_spec(center: np.ndarray, r: float, hinge: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return hinge pivot and axis; user passes hinge position + positive angle.

    hinge choices:
    - xp/xn/zp/zn  -> pivot at (+/-r,0,0)/(0,0,+/-r) around center.
    """
    if hinge == "xp":
        pivot = center + np.array([r, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif hinge == "xn":
        pivot = center + np.array([-r, 0.0, 0.0], dtype=np.float64)
        axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    elif hinge == "zp":
        pivot = center + np.array([0.0, 0.0, r], dtype=np.float64)
        axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    elif hinge == "zn":
        pivot = center + np.array([0.0, 0.0, -r], dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown hinge '{hinge}', expected one of xp/xn/zp/zn")
    return pivot, axis


def rotate_lid_positive_angle(
    lid_closed: trimesh.Trimesh,
    body_center: np.ndarray,
    body_r: float,
    hinge: str,
    angle_deg: float,
) -> trimesh.Trimesh:
    pivot, axis = hinge_spec(body_center, body_r, hinge)
    angle = np.deg2rad(float(angle_deg))

    # Use positive angle API, then auto-fix axis sign to ensure "opening upward".
    R = rotation_around_axis(axis, angle)
    v = lid_closed.vertices - pivot
    v_try = v @ R.T + pivot
    if np.mean(v_try[:, 1]) < np.mean(lid_closed.vertices[:, 1]):
        axis = -axis
        R = rotation_around_axis(axis, angle)
        v_try = v @ R.T + pivot

    out = lid_closed.copy()
    out.vertices = v_try
    return out


def preprocess_body(mesh: trimesh.Trimesh, target_size: float = 0.26, body_y_offset: float = -0.09) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(-out.centroid)
    s = target_size / max(np.max(out.extents), 1e-8)
    out.apply_scale(s)
    out.apply_translation([0.0, body_y_offset, 0.0])
    return out


def rig_teapot_pair(
    body_path: Path,
    lid_path: Path,
    out_dir: Path,
    open_angles: Tuple[float, ...] = (0.0, 20.0, 40.0, 60.0),
    hinge: str = "xp",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    body_raw = trimesh.load_mesh(str(body_path))
    lid_raw = trimesh.load_mesh(str(lid_path))

    body = preprocess_body(body_raw)
    body_inner = detect_body_top_inner_ring(body)

    lid_local, lid_base_circle, lid_r = preprocess_lid_to_local_base(lid_raw)
    lid_closed, scale = align_lid_to_body_inner_ring(lid_local, lid_r, body_inner)

    body.export(out_dir / "body_aligned.obj")
    lid_closed.export(out_dir / "lid_aligned_closed.obj")

    closed_pair = trimesh.util.concatenate([body, lid_closed])
    closed_pair.export(out_dir / "teapot_closed.obj")

    angle_files = []
    for a in open_angles:
        lid_open = rotate_lid_positive_angle(
            lid_closed=lid_closed,
            body_center=body_inner.center,
            body_r=body_inner.radius,
            hinge=hinge,
            angle_deg=float(a),
        )
        pair_open = trimesh.util.concatenate([body, lid_open])
        name = f"teapot_open_{int(round(a)):03d}.obj"
        pair_open.export(out_dir / name)
        angle_files.append({"angle_deg": float(a), "mesh": name})

    meta: Dict[str, object] = {
        "body_mesh": str(body_path),
        "lid_mesh": str(lid_path),
        "algorithm": {
            "lid_base": "diameter from X/Z extents, O translated to y=0",
            "body_ring": "top inner ring from watertight top band",
            "alignment": "align O to body inner center, scale r to body inner radius",
            "hinge_api": "hinge position + positive angle",
        },
        "fit": {
            "lid_radius_before": float(lid_base_circle.radius),
            "body_inner_radius": float(body_inner.radius),
            "scale": float(scale),
            "body_inner_center": body_inner.center.tolist(),
            "hinge": hinge,
        },
        "outputs": {
            "body": "body_aligned.obj",
            "lid_closed": "lid_aligned_closed.obj",
            "closed_pair": "teapot_closed.obj",
            "open_pairs": angle_files,
        },
    }

    with open(out_dir / "rig_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Auto rig done.")
    print("Output dir:", out_dir)
    print("Fit:", json.dumps(meta["fit"], indent=2))


def parse_angles(s: str) -> Tuple[float, ...]:
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if not vals:
        return (0.0,)
    return tuple(float(v) for v in vals)


def main():
    parser = argparse.ArgumentParser(description="Rewrite: teapot lid-body alignment with explicit hinge API.")
    parser.add_argument("--body", type=str, required=True, help="Path to teapot body mesh (.obj)")
    parser.add_argument("--lid", type=str, required=True, help="Path to lid mesh (.obj)")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--angles",
        type=str,
        default="0,20,40,60",
        help="Comma-separated open angles in degrees (positive values).",
    )
    parser.add_argument(
        "--hinge",
        type=str,
        default="xp",
        choices=["xp", "xn", "zp", "zn"],
        help="Hinge pivot position on body ring: xp/xn/zp/zn",
    )

    args = parser.parse_args()

    rig_teapot_pair(
        body_path=Path(args.body),
        lid_path=Path(args.lid),
        out_dir=Path(args.out),
        open_angles=parse_angles(args.angles),
        hinge=args.hinge,
    )


if __name__ == "__main__":
    main()
