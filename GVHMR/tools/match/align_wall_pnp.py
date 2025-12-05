#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import json
import math
from typing import Tuple, List, Optional, Iterable

import numpy as np
import cv2
from PIL import Image
import trimesh

# =========================
# ZoeDepth hub camera intrinsics (FOV=55°, principal point at center)
# image need to transform to thumbnail((1024,1024)) 
# =========================
def zoe_hub_intrinsics(H: int, W: int, fov_deg: float = 55.0) -> np.ndarray:
    f = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
    K = np.array([[f, 0, W * 0.5],
                  [0, f, H * 0.5],
                  [0, 0, 1.0]], dtype=np.float32)
    return K

def zoe_M_flip() -> np.ndarray:
    # ZoeDepth depth_to_points to pytorch3D
    return np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

# =========================
# IO
# =========================
def load_hub_image_for_zoe(path: str, max_side: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))  # scale
    return np.array(img)  # H,W,3 uint8

def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded mesh is not a Trimesh.")
    return mesh

# =========================
# Feature detection & matching
# =========================
def detect_and_match(img0: np.ndarray, img1: np.ndarray,
                     max_kp: int = 4000, ratio: float = 0.75):
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_kp)
        kp0, des0 = detector.detectAndCompute(gray0, None)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn = flann.knnMatch(des0, des1, k=2)
    else:
        detector = cv2.ORB_create(nfeatures=max_kp)
        kp0, des0 = detector.detectAndCompute(gray0, None)
        kp1, des1 = detector.detectAndCompute(gray1, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des0, des1, k=2)

    matches = []
    for m in knn:
        if len(m) < 2:
            continue
        m1, m2 = m
        if m1.distance < ratio * m2.distance:
            matches.append(m1)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    return pts0, pts1, matches, kp0, kp1

def ransac_homography_filter(pts0: np.ndarray, pts1: np.ndarray,
                             reproj_thresh: float = 3.0):
    if len(pts0) < 8:
        return None, None
    H, inlier = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC, reproj_thresh, confidence=0.999)
    if inlier is None:
        return None, None
    return H, inlier.reshape(-1).astype(bool)

# =========================
# ZoeDepth pixel -> vertex index
# =========================
def pixel_to_vertex_index(u: float, v: float, W: int, H: int) -> Optional[int]:
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= W or y < 0 or y >= H:
        return None
    return y * W + x

# =========================
# PnP
# =========================
def solve_pnp(X3d: np.ndarray, x2d: np.ndarray, K: np.ndarray):
    if len(X3d) < 4:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X3d.astype(np.float32), x2d.astype(np.float32),
        K.astype(np.float32), distCoeffs=None,
        iterationsCount=2000, reprojectionError=2.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inliers is None or len(inliers) < 4:
        return None
    inliers = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        X3d[inliers].astype(np.float32),
        x2d[inliers].astype(np.float32),
        K.astype(np.float32), None, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t, inliers, rvec, tvec

def compute_frame_intrinsics(frame_img: np.ndarray,
                             fx: Optional[float], fy: Optional[float],
                             cx: Optional[float], cy: Optional[float],
                             fov_deg: Optional[float]) -> np.ndarray:
    H, W = frame_img.shape[:2]
    if fx is not None and fy is not None and cx is not None and cy is not None:
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
    elif fov_deg is not None:
        f = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
        K = np.array([[f, 0, W * 0.5],
                      [0, f, H * 0.5],
                      [0, 0, 1]], dtype=np.float32)
    else:
        # fallback：用 55° 當估計
        f = 0.5 * W / math.tan(0.5 * math.radians(55.0))
        K = np.array([[f, 0, W * 0.5],
                      [0, f, H * 0.5],
                      [0, 0, 1]], dtype=np.float32)
    return K

# =========================
# Video iterator
# =========================
def iter_frames_from_video(video_path: str) -> Iterable[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        yield idx, frame_bgr
        idx += 1
    cap.release()

# =========================
# Visualization
# =========================
def draw_inliers(img0, img1, kp0_xy, kp1_xy, inlier_mask, out_path):
    c0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    c1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    H0, W0 = c0.shape[:2]
    H1, W1 = c1.shape[:2]
    target_h = max(H0, H1)
    scale0 = target_h / float(H0)
    scale1 = target_h / float(H1)
    new_w0 = int(round(W0 * scale0))
    new_w1 = int(round(W1 * scale1))
    c0r = cv2.resize(c0, (new_w0, target_h), interpolation=cv2.INTER_LINEAR)
    c1r = cv2.resize(c1, (new_w1, target_h), interpolation=cv2.INTER_LINEAR)
    kp0s = kp0_xy.copy()
    kp1s = kp1_xy.copy()
    kp0s[:, 0] *= scale0  # x
    kp0s[:, 1] *= scale0  # y
    kp1s[:, 0] *= scale1
    kp1s[:, 1] *= scale1

    canvas = np.hstack([c0r, c1r])
    x_offset = c0r.shape[1]

    for i, ok in enumerate(inlier_mask):
        if not ok:
            continue
        if i >30: 
            break
        u0, v0 = kp0s[i]
        u1, v1 = kp1s[i]
        p0 = (int(round(u0)), int(round(v0)))
        p1 = (int(round(u1)) + x_offset, int(round(v1)))
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.line(canvas, p0, p1, (0, 255, 0), 1)

    cv2.imwrite(out_path, canvas)



def main():
    ap = argparse.ArgumentParser(description="Align frames (folder or video) to ZoeDepth hub mesh via PnP.")
    ap.add_argument("--hub-image", required=True, help="Single-view wall image used for ZoeDepth.")
    ap.add_argument("--mesh", required=True, help="Wall mesh (GLB/PLY) from ZoeDepth.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", help="Directory of frames.")
    src.add_argument("--video", help="Video file (mp4/avi).")
    ap.add_argument("--glob", default="*.jpg", help="Glob under --frames-dir (default: *.jpg).")
    ap.add_argument("--out", default="poses.csv", help="Output CSV for poses.")
    ap.add_argument("--save-corr", action="store_true", help="Save per-frame inlier correspondences (.npz).")
    ap.add_argument("--save-viz", action="store_true", help="Save per-frame inlier visualization.")
    ap.add_argument("--frame-fov-deg", type=float, default=None, help="If fx/fy/cx/cy not given, use this FOV for frames.")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    args = ap.parse_args()

    # Hub image & intrinsics
    hub_img = load_hub_image_for_zoe(args.hub_image, max_side=1024)
    H0, W0 = hub_img.shape[:2]
    K0 = zoe_hub_intrinsics(H0, W0, fov_deg=55.0)
    R0 = np.eye(3, dtype=np.float32)
    t0 = np.zeros((3, 1), dtype=np.float32)
    M = zoe_M_flip()  # for record

    # Mesh (ZoeDepth: vertex i <-> (u=i%W0, v=i//W0))
    mesh = load_mesh(args.mesh)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    valid_vtx_mask = np.isfinite(V).all(axis=1)
    if V.shape[0] < (H0 * W0) * 0.8:
        print(f"[WARN] Vertex count {V.shape[0]} << H*W={H0*W0}. Some hub pixels may not map to a vertex (edge filtering).")

    # Prepare output CSV
    with open(args.out, "w") as f:
        cols = [
            "frame",
            "rvec_x","rvec_y","rvec_z",
            "tvec_x","tvec_y","tvec_z",
            "R00","R01","R02","R10","R11","R12","R20","R21","R22",
            "t0","t1","t2",
            "num_inliers","num_corr"
        ]
        f.write(",".join(cols) + "\n")

    # Source iterator (images or video)
    if args.frames_dir is not None:
        if not os.path.isdir(args.frames_dir):
            raise RuntimeError(f"--frames-dir not a directory: {args.frames_dir}")
        frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, args.glob)))
        if not frame_paths:
            raise RuntimeError("No frames found under --frames-dir with given --glob.")
        iterable = enumerate(frame_paths)  # yields (idx, path)
        mode = "images"
    else:
        if not os.path.isfile(args.video):
            raise RuntimeError(f"Video not found: {args.video}")
        iterable = iter_frames_from_video(args.video)  # yields (idx, frame_bgr)
        mode = "video"

    # Process frames
    for idx, item in iterable:
        if mode == "images":
            fp = item
            frame_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print(f"[WARN] Cannot read frame: {fp}")
                continue
            name = os.path.basename(fp)
        else:
            frame_bgr = item
            name = f"frame_{idx:06d}.jpg"

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        Kt = compute_frame_intrinsics(frame_rgb, args.fx, args.fy, args.cx, args.cy, args.frame_fov_deg)

        # Feature matching hub <-> frame
        pts0, pts1, matches, kp0, kp1 = detect_and_match(hub_img, frame_rgb)
        if len(pts0) < 20:
            print(f"[{name}] Too few matches: {len(pts0)}")
            continue

        # Homography filter (optional but recommended)
        H, inlier_h = ransac_homography_filter(pts0, pts1, reproj_thresh=3.0)
        if inlier_h is None:
            print(f"[{name}] Homography failed.")
            continue
        pts0_f = pts0[inlier_h]
        pts1_f = pts1[inlier_h]

        # Lift to 3D using ZoeDepth vertex indexing
        X3d = []
        x2d = []
        for (u0, v0), (ut, vt) in zip(pts0_f, pts1_f):
            vi = pixel_to_vertex_index(u0, v0, W0, H0)
            if vi is None or vi >= V.shape[0]:
                continue
            if not valid_vtx_mask[vi]:
                continue
            X = V[vi]
            if not np.isfinite(X).all():
                continue
            X3d.append(X)
            x2d.append([ut, vt])

        X3d = np.asarray(X3d, dtype=np.float32)
        x2d = np.asarray(x2d, dtype=np.float32)
        if len(X3d) < 6:
            print(f"[{name}] Too few 3D-2D pairs after lifting: {len(X3d)}")
            continue

        # PnP
        result = solve_pnp(X3d, x2d, Kt)
        if result is None:
            print(f"[{name}] PnP failed.")
            continue
        R, t, inliers, rvec, tvec = result

        # Save correspondences
        if args.save_corr:
            np.savez_compressed(
                f"corr_{os.path.splitext(name)[0]}.npz",
                X3d=X3d[inliers], x2d=x2d[inliers], Kt=Kt, R=R, t=t
            )

        # Save visualization (draw all homography inliers for simplicity)
        if args.save_viz:
            inlier_mask = np.ones(len(pts0_f), dtype=bool)
            out_png = f"match_result/viz_{os.path.splitext(name)[0]}.png"
            draw_inliers(hub_img, frame_rgb, pts0_f, pts1_f, inlier_mask, out_png)

        # Append CSV row
        with open(args.out, "a") as f:
            row = [name]
            row += list(map(str, rvec.reshape(-1).tolist()))
            row += list(map(str, tvec.reshape(-1).tolist()))
            row += list(map(str, R.reshape(-1).tolist()))
            row += list(map(str, t.reshape(-1).tolist()))
            row += [str(len(inliers)), str(len(X3d))]
            f.write(",".join(row) + "\n")

        print(f"[{name}] OK: inliers={len(inliers)}/{len(X3d)}")

    # Save hub camera meta for reuse
    hub_meta = {
        "hub_image": os.path.abspath(args.hub_image),
        "hub_size_hw": [int(H0), int(W0)],
        "K0": K0.tolist(),
        "R0": R0.tolist(),
        "t0": t0.reshape(-1).tolist(),
        "M_flip": zoe_M_flip().tolist(),
        "note": "ZoeDepth hub camera: FOV=55deg, principal at center; R0=I, t0=0 in PyTorch3D-style coords."
    }
    with open("hub_camera.json", "w") as jf:
        json.dump(hub_meta, jf, indent=2)
    print("Saved hub_camera.json and poses to", args.out)

if __name__ == "__main__":
    main()
