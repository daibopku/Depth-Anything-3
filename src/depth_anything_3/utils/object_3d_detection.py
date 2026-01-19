"""Utilities for deriving 3D object bounding boxes from video + mask inputs.

This module builds on Depth Anything 3 predictions (depth + camera poses) to:
- back-project masked pixels into a world-space point cloud
- estimate a stable orientation (x/y from PCA on the XY plane, z from world up)
- compute per-frame oriented boxes (center + size) for each surviving object
- remap masks to the filtered/sorted object IDs

The primary entry point is :func:`compute_3d_bboxes_from_prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2

import numpy as np

from depth_anything_3.specs import Prediction


@dataclass
class Object3DBoundingBoxes:
    """Container for 3D bounding box results.

    Attributes:
        bboxes: Array shaped ``(F, N, 6)`` where each box is ``(cx, cy, cz, sx, sy, sz)``
            expressed in world coordinates. ``F`` = number of frames, ``N`` = number of
            surviving objects after filtering. If an object is absent in a frame, the
            corresponding row is ``0``.
        mask: Remapped mask with shape ``(F, H, W)`` where object IDs are re-indexed to
            ``1..N`` (0 keeps background). ``H``/``W`` follow the input mask size.
        id_mapping: Mapping from original object id -> new id (1-based).
    counts: Total surviving 3D point count per **new** id.
    axes: Per-object orientation bases (N, 3, 3) where rows are x/y/z unit vectors.
        intrinsics: Camera intrinsics aligned with depth/output resolution (N, 3, 3).
        extrinsics_w2c: World-to-camera extrinsics (N, 4, 4 or 3,4).
        extrinsics_c2w: Camera-to-world extrinsics (N, 4, 4).
        image_size: (H, W) of the depth / processed outputs used for boxes.
        mask_size: (H, W) of the returned ``mask`` (original mask resolution).
    """

    bboxes: np.ndarray
    mask: np.ndarray
    id_mapping: Dict[int, int]
    counts: Dict[int, int]
    axes: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    extrinsics_w2c: np.ndarray | None = None
    extrinsics_c2w: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    mask_size: tuple[int, int] | None = None
    depth: np.ndarray | None = None
    conf: np.ndarray | None = None
    fps: float | None = None


def compute_3d_bboxes_from_prediction(
    prediction: Prediction,
    mask: np.ndarray,
    *,
    min_points: int = 200,
    conf_thresh: float = 1.05,
) -> Object3DBoundingBoxes:
    """Compute oriented 3D bounding boxes from a DA3 prediction and pixel masks.

    Args:
        prediction: Depth Anything 3 ``Prediction`` containing depth/intrinsics/extrinsics.
        mask: Integer mask of shape ``(F, H, W)`` whose values are object ids (0=background).
        min_points: Minimum accumulated 3D points (across all frames) required to keep an
            object. Objects below this threshold are removed from the outputs.
        conf_thresh: Confidence threshold applied before back-projecting depth.

    Returns:
        ``Object3DBoundingBoxes`` with ``bboxes`` shape ``(F, N, 6)`` and remapped ``mask``.
        Boxes are ordered by descending total point count.
    """

    depth = prediction.depth
    intrinsics = prediction.intrinsics
    extrinsics = prediction.extrinsics
    conf = prediction.conf

    if depth is None or intrinsics is None or extrinsics is None:
        raise ValueError("prediction must contain depth, intrinsics, and extrinsics")
    if mask.ndim != 3:
        raise ValueError(f"mask must be (F,H,W), got shape {mask.shape}")
    if depth.shape[0] != mask.shape[0]:
        raise ValueError("depth and mask must have the same frame count")

    frames, H, W = depth.shape
    mask_orig = mask.astype(np.int64, copy=False)
    orig_H, orig_W = mask_orig.shape[1:]

    # If mask spatial resolution differs from model output (due to resize/crop in preprocessing),
    # align mask to depth resolution using nearest-neighbor to preserve labels (processing only).
    mask_proc = mask_orig
    if mask_proc.shape[1:] != (H, W):
        resized = []
        for f in range(frames):
            resized_mask = cv2.resize(mask_proc[f].astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            resized.append(resized_mask)
        mask_proc = np.stack(resized, axis=0)

    c2w_list = _invert_extrinsics(extrinsics)

    per_frame_points: List[Dict[int, np.ndarray]] = []
    global_points: Dict[int, List[np.ndarray]] = {}
    total_counts: Dict[int, int] = {}

    for f in range(frames):
        frame_points = _collect_frame_points(
            depth[f], intrinsics[f], c2w_list[f], mask_proc[f], conf[f] if conf is not None else None, conf_thresh
        )
        per_frame_points.append(frame_points)
        for obj_id, pts in frame_points.items():
            if obj_id == 0 or pts.size == 0:
                continue
            if obj_id not in global_points:
                global_points[obj_id] = []
            global_points[obj_id].append(pts)
            total_counts[obj_id] = total_counts.get(obj_id, 0) + pts.shape[0]

    # Filter objects by min_points and sort by total point count (desc)
    kept_ids = [oid for oid, cnt in total_counts.items() if cnt >= min_points and oid != 0]
    kept_ids.sort(key=lambda oid: total_counts[oid], reverse=True)

    # Re-index kept objects to contiguous ids starting from 1; 0 is always background
    id_mapping: Dict[int, int] = {oid: new_id for new_id, oid in enumerate(kept_ids, start=1)}
    counts_new: Dict[int, int] = {id_mapping[oid]: 0 for oid in kept_ids}

    new_mask_proc = np.zeros_like(mask_proc, dtype=np.int32)
    for oid, nid in id_mapping.items():
        new_mask_proc[mask_proc == oid] = nid

    # Map remapped mask back to the original spatial resolution
    if (orig_H, orig_W) != (H, W):
        new_mask = np.zeros_like(mask_orig, dtype=np.int32)
        for f in range(frames):
            new_mask[f] = cv2.resize(
                new_mask_proc[f].astype(np.int32), (orig_W, orig_H), interpolation=cv2.INTER_NEAREST
            )
    else:
        new_mask = new_mask_proc

    # Recompute per-frame points for kept objects using only mask/depth (no confidence filter)
    per_frame_points_kept: List[Dict[int, np.ndarray]] = []
    for f in range(frames):
        frame_points_full = _collect_frame_points(
            depth[f], intrinsics[f], c2w_list[f], mask_proc[f], None, conf_thresh
        )
        frame_kept: Dict[int, np.ndarray] = {}
        for oid in kept_ids:
            pts = frame_points_full.get(oid)
            if pts is None or pts.shape[0] == 0:
                continue
            frame_kept[oid] = pts
            counts_new[id_mapping[oid]] += pts.shape[0]
        per_frame_points_kept.append(frame_kept)

    num_objects = len(kept_ids)
    bboxes = np.zeros((frames, num_objects, 6), dtype=np.float32)

    # Use world-aligned axes for all objects (avoid PCA-based orientation to stay stable)
    axes_by_oid: Dict[int, np.ndarray] = {oid: np.eye(3, dtype=np.float32) for oid in kept_ids}

    axes_array = (
        np.stack([axes_by_oid[oid] for oid in kept_ids], axis=0) if len(kept_ids) > 0 else np.zeros((0, 3, 3), dtype=np.float32)
    )

    for f in range(frames):
        for oid in kept_ids:
            pts = per_frame_points_kept[f].get(oid)
            if pts is None or pts.shape[0] == 0:
                continue
            pts = _filter_outliers(pts)
            if pts.shape[0] == 0:
                continue
            axes = axes_by_oid[oid]
            bbox = _compute_bbox_from_points(pts, axes)
            bboxes[f, id_mapping[oid] - 1] = bbox

    return Object3DBoundingBoxes(
        bboxes=bboxes,
        mask=new_mask,
        id_mapping=id_mapping,
        counts=counts_new,
        axes=axes_array,
        intrinsics=prediction.intrinsics,
        extrinsics_w2c=prediction.extrinsics,
        extrinsics_c2w=np.stack(c2w_list, axis=0),
        image_size=(H, W),
        mask_size=(orig_H, orig_W),
        depth=prediction.depth,
        conf=prediction.conf,
    )


def _invert_extrinsics(extrinsics: np.ndarray) -> List[np.ndarray]:
    """Convert ``w2c`` extrinsics to homogeneous ``c2w`` matrices."""

    exts = []
    for ext in extrinsics:
        ext44 = _as_homogeneous44(ext)
        exts.append(np.linalg.inv(ext44).astype(np.float32))
    return exts


def _collect_frame_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray | None,
    conf_thresh: float,
) -> Dict[int, np.ndarray]:
    """Back-project the valid masked pixels of one frame into world space."""

    # Do not assume a background id (e.g., mask==0). Keep all labels here and rely on
    # point-count filtering later to discard tiny/noisy objects.
    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        valid &= conf >= conf_thresh

    if not np.any(valid):
        return {}

    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pixels = np.stack([us[valid], vs[valid], np.ones_like(us[valid])], axis=0)  # (3, M)

    depth_vals = depth[valid].astype(np.float32)
    ids = mask[valid].astype(np.int64)

    K_inv = np.linalg.inv(intrinsic).astype(np.float32)
    rays = K_inv @ pixels  # (3, M)
    Xc = rays * depth_vals[None, :]  # (3, M)
    Xc_h = np.concatenate([Xc, np.ones((1, Xc.shape[1]), dtype=np.float32)], axis=0)  # (4, M)
    Xw = (c2w @ Xc_h)[:3].T  # (M, 3)

    points_by_obj: Dict[int, np.ndarray] = {}
    for oid in np.unique(ids):
        pts = Xw[ids == oid]
        points_by_obj[int(oid)] = pts.astype(np.float32)
    return points_by_obj


def _filter_outliers(points: np.ndarray, quantile: float = 0.01, min_points: int = 5) -> np.ndarray:
    """Remove NaNs/Infs and clip spatial outliers per axis by quantiles.

    Args:
        points: (N, 3) array.
        quantile: Lower/upper quantile to drop (e.g., 0.01 keeps [1%, 99%]).
        min_points: If remaining points fall below this, return empty to skip bbox.
    """

    if points.size == 0:
        return points

    pts = points[np.isfinite(points).all(axis=1)]
    if pts.shape[0] == 0:
        return pts

    q_low = np.quantile(pts, quantile, axis=0)
    q_high = np.quantile(pts, 1.0 - quantile, axis=0)
    inliers = (pts >= q_low) & (pts <= q_high)
    mask = inliers.all(axis=1)
    pts = pts[mask]

    if pts.shape[0] < min_points:
        return np.empty((0, 3), dtype=points.dtype)
    return pts


def _estimate_axes_xy_pca(points: np.ndarray | None) -> np.ndarray:
    """Estimate orthonormal axes: x from XY-PCA, y = z×x, z = world up."""

    if points is None or points.shape[0] < 2:
        return np.eye(3, dtype=np.float32)

    xy = points[:, :2].astype(np.float32)
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    x_xy = eigvecs[:, order[0]]  # 2-dim

    x_axis = np.array([x_xy[0], x_xy[1], 0.0], dtype=np.float32)
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis /= norm_x

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = np.cross(z_axis, x_axis)
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis /= norm_y

    return np.stack([x_axis, y_axis, z_axis], axis=0)


def _compute_bbox_from_points(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Compute oriented bounding box (center + size) in world space."""

    if points.shape[0] == 0:
        return np.zeros(6, dtype=np.float32)

    # Project to local coords defined by axes rows
    local = points @ axes.T  # (N,3)
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    center_local = (mins + maxs) * 0.5
    size = (maxs - mins)
    center_world = center_local @ axes
    return np.concatenate([center_world.astype(np.float32), size.astype(np.float32)])


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """Ensure extrinsic is 4x4 homogeneous."""

    if ext.shape == (4, 4):
        return ext.astype(np.float32)
    if ext.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = ext.astype(np.float32)
        return out
    raise ValueError(f"Unsupported extrinsic shape: {ext.shape}")