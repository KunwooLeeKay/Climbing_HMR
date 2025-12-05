#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json
from pathlib import Path
import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- IO ----------
def load_poses_csv(csv_path):
    poses = {}
    with open(csv_path, "r") as f:
        r = csv.reader(f); _ = next(r)
        for row in r:
            name = row[0]
            R = np.array(list(map(float, row[7:16])), dtype=np.float32).reshape(3,3)
            t = np.array(list(map(float, row[16:19])), dtype=np.float32).reshape(3,1)
            poses[name] = {"R": R, "t": t}
    return poses

class SMPLIncamCache:
    def __init__(self, cache_dir: str):
        cd = Path(cache_dir)
        self.V = np.load(cd / "verts_incam.npy", mmap_mode="r")
        self.J = np.load(cd / "joints_incam.npy", mmap_mode="r")
        self.meta = json.load(open(cd / "meta.json", "r"))
        self.L = int(self.meta["L"])
    @staticmethod
    def idx_from_name(name: str) -> int:
        stem = Path(name).stem
        digs = "".join([c for c in stem if c.isdigit()])
        if digs == "": raise ValueError(f"Cannot parse frame index from {name}")
        return int(digs)
    def load_frame(self, name: str):
        i = self.idx_from_name(name)
        return np.asarray(self.V[i]), np.asarray(self.J[i])

# ---------- geometry ----------
def cam_to_wall(X_cam, R, t):
    return (X_cam - t.reshape(1,3)) @ R.T

def umeyama_similarity(X, Y, with_scaling=True, with_rotation=True):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    muX = X.mean(axis=0); muY = Y.mean(axis=0)
    Xc = X - muX; Yc = Y - muY
    if with_rotation:
        C = (Yc.T @ Xc) / X.shape[0]
        U, D, Vt = np.linalg.svd(C)
        S = np.eye(3); 
        if np.linalg.det(U @ Vt) < 0: S[2,2] = -1.0
        R = U @ S @ Vt
    else:
        R = np.eye(3)
    if with_scaling:
        varX = (Xc**2).sum() / X.shape[0]
        s = np.trace((Yc.T @ (Xc @ R.T))) / (X.shape[0] * varX)
    else:
        s = 1.0
    t = muY - s * (R @ muX)
    return s, R, t

def rot_angle_deg(R):
    tr = np.clip((np.trace(R) - 1)/2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def export_combined_obj(out_path: Path, wall_V, wall_F, smpl_V, smpl_F=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# combined wall + smpl (wall coordinates)\n")
        f.write("g wall\n")
        for v in wall_V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        base = wall_V.shape[0]
        if wall_F is not None and len(wall_F)>0:
            for a,b,c in wall_F.astype(int): f.write(f"f {a+1} {b+1} {c+1}\n")
        f.write("g smpl\n")
        for v in smpl_V: f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if smpl_F is not None and len(smpl_F)>0:
            for a,b,c in smpl_F.astype(int): f.write(f"f {a+1+base} {b+1+base} {c+1+base}\n")
        else:
            for i in range(smpl_V.shape[0]): f.write(f"p {base + i + 1}\n")


def nearest_on_vertices(points, verts):
    if HAVE_SCIPY:
        tree = cKDTree(verts)
        dist, idx = tree.query(points, k=1)
        return verts[idx], dist
    diffs = points[:,None,:] - verts[None,:,:]
    dist2 = np.sum(diffs*diffs, axis=2)
    idx = np.argmin(dist2, axis=1)
    dist = np.sqrt(dist2[np.arange(points.shape[0]), idx])
    return verts[idx], dist


def main():
    ap = argparse.ArgumentParser(description="Fit global similarity (Sim(3) variants) to align SMPL to wall.")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--smpl-cache", required=True)
    ap.add_argument("--wall-mesh", required=True)
    ap.add_argument("--smpl-faces-npy", default=None)
    ap.add_argument("--frames-list", default=None)
    ap.add_argument("--sample-frames", type=int, default=80)
    ap.add_argument("--contact-joints", nargs="*", type=int, default=[4,7,10,13])
    ap.add_argument("--nn-per-frame", type=int, default=200)
    ap.add_argument("--cap-seq", default="8,4,2,1",
                    help="comma-separated distance caps (meters) to try in order")
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="stop when at least this many correspondences are collected")
    ap.add_argument("--trim-frac", type=float, default=0.5,
                    help="keep closest fraction before fitting (0<trim<=1)")
    ap.add_argument("--fit-mode", choices=["sim3","scale_trans","scale_only"], default="scale_trans")
    ap.add_argument("--export-check", action="store_true")
    ap.add_argument("--export-dir", default="export_objs_aligned")
    args = ap.parse_args()

    poses = load_poses_csv(args.poses)
    smpl = SMPLIncamCache(args.smpl_cache)
    wall = trimesh.load(args.wall_mesh, process=False)
    if isinstance(wall, trimesh.Scene):
        wall = trimesh.util.concatenate(tuple(wall.geometry.values()))
    wall_V = np.asarray(wall.vertices, dtype=np.float32)
    wall_F = np.asarray(wall.faces, dtype=np.int32) if wall.faces is not None else None
    wall_prox = ProximityQuery(wall)

    smpl_F = None
    if args.smpl_faces_npy is not None:
        smpl_F = np.load(args.smpl_faces_npy).astype(np.int32)

    # frame list
    if args.frames_list:
        names = [ln.strip() for ln in open(args.frames_list) if ln.strip() in poses]
    else:
        names = list(poses.keys())
    names = sorted(names)[:args.sample_frames]
    if not names:
        raise RuntimeError("No frames selected; check poses.csv and --frames-list")

    print(f"[info] frames used: {len(names)}; contact joints: {args.contact_joints}")

    caps = [float(x) for x in args.cap-seq.split(",")] if hasattr(args, "cap-seq") else [8,4,2,1]
    # argparse doesn't like '-' in attr; fallback robustly:
    try:
        caps = [float(x) for x in args.cap_seq.split(",")]
    except Exception:
        pass

    X_best = Y_best = D_best = None
    used_cap = None
    for cap in caps:
        X_smpl, Y_wall, D = [], [], []
        total_pairs = 0
        max_pairs_total = args.nn_per_frame * len(names)

        for name in names:
            R, t = poses[name]["R"], poses[name]["t"]
            V_cam, J_cam = smpl.load_frame(name)

            # joints first
            if len(args.contact_joints) > 0 and max(args.contact_joints) < J_cam.shape[0]:
                Js = J_cam[args.contact_joints]
            else:
                Js = J_cam
            Js_w = cam_to_wall(Js, R, t)

            P, d, tri = wall_prox.on_surface(Js_w)
            m = d < cap
            if m.any():
                X_smpl.append(Js_w[m]); Y_wall.append(P[m]); D.append(d[m])
                total_pairs += int(m.sum())

            # always add a vertex-based batch as well (helps when joints are away)
            nv = min(1200, V_cam.shape[0])
            idx = np.linspace(0, V_cam.shape[0]-1, nv, dtype=int)
            Vs_w = cam_to_wall(V_cam[idx], R, t)
            Q, dist_v = nearest_on_vertices(Vs_w, wall_V)
            m2 = dist_v < cap
            if m2.any():
                X_smpl.append(Vs_w[m2]); Y_wall.append(Q[m2]); D.append(dist_v[m2])
                total_pairs += int(m2.sum())

            if total_pairs >= max_pairs_total:
                break

        if len(X_smpl) == 0:
            print(f"[info] cap={cap} → 0 pairs")
            continue

        X = np.concatenate(X_smpl, axis=0)
        Y = np.concatenate(Y_wall, axis=0)
        Dist = np.concatenate(D, axis=0)
        print(f"[info] cap={cap} → collected {len(X)} pairs (median {np.median(Dist):.3f} m)")

        if len(X) >= args.min_pairs:
            X_best, Y_best, D_best, used_cap = X, Y, Dist, cap
            break

        # keep the best so far, maybe next cap is smaller (we go high->low, but just in case)
        if X_best is None or len(X) > len(X_best):
            X_best, Y_best, D_best, used_cap = X, Y, Dist, cap

    if X_best is None or len(X_best) < 4:
        raise RuntimeError("No correspondences collected even with widest cap.")

    # trim to closest
    trim = np.clip(args.trim_frac, 1e-3, 1.0)
    k = max(4, int(trim * len(X_best)))
    sel = np.argsort(D_best)[:k]
    X = X_best[sel]; Y = Y_best[sel]; Dist = D_best[sel]

    print(f"[info] using cap={used_cap}, kept {len(X)} pairs after trim "
          f"(median prefit dist={np.median(Dist):.3f} m)")

    # fit
    with_rot = (args.fit_mode == "sim3")
    with_s = (args.fit_mode in ["sim3","scale_trans","scale_only"])
    if args.fit_mode == "scale_only":
        with_rot = False
    s, Rdelta, tdelta = umeyama_similarity(X, Y, with_scaling=with_s, with_rotation=with_rot)
    ang = rot_angle_deg(Rdelta)
    res = np.linalg.norm((s * (X @ Rdelta.T) + tdelta) - Y, axis=1)
    print(f"[Fit] mode={args.fit_mode}  cap={used_cap}  s={s:.4f}  RΔ_angle={ang:.2f}°  "
          f"residual median={np.median(res):.4f}, p90={np.percentile(res,90):.4f}")

    # export a few frames
    if args.export_check:
        outdir = Path(args.export_dir)
        picks = names[::max(1, len(names)//5)] if len(names) > 0 else names
        for name in picks:
            V_cam, _ = smpl.load_frame(name)
            R, t = poses[name]["R"], poses[name]["t"]
            V_wall = cam_to_wall(V_cam, R, t)
            V_aligned = (s * (Rdelta @ V_wall.T)).T + tdelta.reshape(1,3)
            export_combined_obj(outdir / f"combined_{Path(name).stem}.obj",
                                wall_V, wall_F, V_aligned, smpl_F)
        print(f"[export] wrote {len(picks)} OBJs to {args.export_dir}")

    # save
    Path("smpl_wall_similarity.json").write_text(json.dumps({
        "scale": float(s),
        "Rdelta": Rdelta.tolist(),
        "tdelta": tdelta.tolist(),
        "fit_mode": args.fit_mode,
        "trim_frac": float(trim),
        "used_cap": float(used_cap),
        "contact_joints": args.contact_joints
    }, indent=2))
    print("[save] smpl_wall_similarity.json")

if __name__ == "__main__":
    main()
