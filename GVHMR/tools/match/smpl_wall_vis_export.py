import argparse, csv, json, math, os
from pathlib import Path
import numpy as np
import cv2
import trimesh
import json
def load_sim3(json_path="smpl_wall_similarity.json"):
    if not Path(json_path).exists():
        return None
    d = json.loads(Path(json_path).read_text())
    s = float(d["scale"])
    R = np.array(d["Rdelta"], dtype=np.float32)
    t = np.array(d["tdelta"], dtype=np.float32)
    return {"s": s, "R": R, "t": t}

def apply_sim3_wall(X_wall, sim3):
    # X' = s * (R * X + t),：(s * (R @ X^T))^T + t
    s, R, t = sim3["s"], sim3["R"], sim3["t"]
    return (s * (X_wall @ R.T))
    # return (s * (X_wall @ R.T)+t)
    # return (s * (R @ X_wall.T)).T + t.reshape(1,3)


# ----------------- I/O helpers -----------------
def load_poses_csv(csv_path):
    poses = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f); header = next(reader)
        for row in reader:
            name = row[0]
            R = np.array(list(map(float, row[7:16])), dtype=np.float32).reshape(3,3)
            t = np.array(list(map(float, row[16:19])), dtype=np.float32).reshape(3,1)
            poses[name] = {"R": R, "t": t}
    return poses

class SMPLIncamCache:
    def __init__(self, cache_dir: str):
        cd = Path(cache_dir)
        self.V = np.load(cd / "verts_incam.npy", mmap_mode="r")   # (L, Vs, 3)
        self.J = np.load(cd / "joints_incam.npy", mmap_mode="r")  # (L, J, 3)
        self.meta = json.load(open(cd / "meta.json", "r"))
        self.L = int(self.meta["L"])
    @staticmethod
    def _idx_from_name(name: str) -> int:
        stem = Path(name).stem
        digs = "".join([c for c in stem if c.isdigit()])
        if digs == "": raise ValueError(f"Cannot parse frame index from {name}")
        return int(digs)
    def load_smpl_for_frame(self, name: str):
        idx = self._idx_from_name(name)
        return np.asarray(self.V[idx]), np.asarray(self.J[idx])

# ----------------- geometry -----------------
def cam_to_wall(X_cam, R, t):   # poses.csv stores wall->camera
    return (X_cam - t.reshape(1,3)) @ R.T

def wall_to_cam(X_wall, R, t):
    return (X_wall @ R.T) + t.reshape(1,3)

def intrinsics_from_fov(frame_hw, fov_deg=None, fx=None, fy=None, cx=None, cy=None):
    H, W = frame_hw
    if fx is not None:
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    if fov_deg is None: fov_deg = 55.0
    f = 0.5 * W / math.tan(0.5*math.radians(fov_deg))
    return np.array([[f,0,W*0.5],[0,f,H*0.5],[0,0,1]], dtype=np.float32)

def project_points_wall(X_wall, R, t, K):
    Xc = wall_to_cam(X_wall, R, t)
    z = Xc[:,2:3]
    uv = Xc[:,:2] / np.maximum(z, 1e-6)
    u = K[0,0]*uv[:,0] + K[0,2]
    v = K[1,1]*uv[:,1] + K[1,2]
    return np.stack([u,v], axis=1), Xc[:,2].flatten()

# ----------------- visualization -----------------
def draw_overlay(dst_bgr, pts_uv, color, radius=1):
    h, w = dst_bgr.shape[:2]
    pts = pts_uv.astype(int)
    mask = (pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
    for (u,v) in pts[mask]:
        cv2.circle(dst_bgr, (u,v), radius, color, -1)

# ----------------- exporters -----------------
def export_combined_obj(out_path: Path,
                        wall_V: np.ndarray, wall_F: np.ndarray,
                        smpl_V: np.ndarray, smpl_F: np.ndarray|None):
    """
    Write a single OBJ containing:
      - g wall: wall_V / wall_F
      - g smpl: smpl_V / smpl_F (if provided) else just points
    All in the same wall coordinate system.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# combined wall + smpl (wall coordinates)\n")
        # wall vertices
        f.write("g wall\n")
        for v in wall_V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        base = wall_V.shape[0]
        # wall faces
        if wall_F is not None and len(wall_F)>0:
            for a,b,c in wall_F.astype(int):
                f.write(f"f {a+1} {b+1} {c+1}\n")
        # smpl vertices
        f.write("g smpl\n")
        for v in smpl_V:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        if smpl_F is not None and len(smpl_F)>0:
            for a,b,c in smpl_F.astype(int):
                f.write(f"f {a+1+base} {b+1+base} {c+1+base}\n")
        else:
            # write points primitive for SMPL if no faces (not all viewers show 'p'; many do)
            for i in range(smpl_V.shape[0]):
                f.write(f"p {base + i + 1}\n")

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(
        description="Overlay SMPL & wall; export per-frame combined OBJ in wall coords.")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--smpl-cache", required=True)
    ap.add_argument("--wall-mesh", required=True)  # ZoeDepth wall in hub coords
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames-dir")
    g.add_argument("--video")
    ap.add_argument("--glob", default="*.jpg")

    # intrinsics
    ap.add_argument("--fx", type=float); ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float); ap.add_argument("--cy", type=float)
    ap.add_argument("--frame-fov-deg", type=float, default=55.0)

    # overlay controls
    ap.add_argument("--out-dir", default="debug_wall_overlay")
    ap.add_argument("--draw-joints", action="store_true")
    ap.add_argument("--draw-smpl-verts", type=int, default=1500,
                    help="subsample count for SMPL verts in overlay")
    ap.add_argument("--draw-wall-verts", type=int, default=2000,
                    help="subsample count for wall verts in overlay")

    # export controls
    ap.add_argument("--export-obj", action="store_true")
    ap.add_argument("--export-dir", default="export_objs")
    ap.add_argument("--export-step", type=int, default=10,
                    help="export every N frames to keep it light")
    ap.add_argument("--smpl-faces-npy", type=str, default=None,
                    help="optional path to a (Fs,3) numpy of SMPL faces")
    ap.add_argument("--smpl-wall-sim3-json", type=str, default=None,
                    help="optional path to a sim3 json file for SMPL to wall transformation")
        
    args = ap.parse_args()

    poses = load_poses_csv(args.poses)
    smpl = SMPLIncamCache(args.smpl_cache)

    # wall mesh (already in wall coords)
    wall = trimesh.load(args.wall_mesh, process=False)
    if isinstance(wall, trimesh.Scene):
        wall = trimesh.util.concatenate(tuple(wall.geometry.values()))
    wall_V = np.asarray(wall.vertices, dtype=np.float32)
    wall_F = np.asarray(wall.faces, dtype=np.int32) if wall.faces is not None else None

    # optional SMPL faces
    smpl_F = None
    if args.smpl_faces_npy is not None:
        smpl_F = np.load(args.smpl_faces_npy).astype(np.int32)

    # frame iteration
    if args.frames_dir is not None:
        frame_paths = sorted(Path(args.frames_dir).glob(args.glob))
        iterator = [("images", str(p)) for p in frame_paths]
        def get_frame(src):
            return cv2.imread(src, cv2.IMREAD_COLOR), os.path.basename(src)
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {args.video}")
        def iterator_gen():
            idx = 0
            while True:
                ok, frm = cap.read()
                if not ok: break
                yield ("video", (idx, frm))
                idx += 1
        iterator = iterator_gen()
        def get_frame(src):
            idx, frame = src
            return frame, f"frame_{idx:06d}.jpg"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.export_dir)

    for k, item in enumerate(iterator):
        mode, src = item
        bgr, name = get_frame(src)
        if bgr is None: continue
        if name not in poses:
            print(name)
            # pose missing for this frame (possible if PnP rejected); skip
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        Kt = intrinsics_from_fov((H,W),
                                 fov_deg=args.frame_fov_deg,
                                 fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
        R = poses[name]["R"]; t = poses[name]["t"]

        # load SMPL incam -> wall
        V_cam, J_cam = smpl.load_smpl_for_frame(name)
        V_wall = cam_to_wall(V_cam, R, t)
        J_wall = cam_to_wall(J_cam, R, t)
        # scaLING
        # sim3 = load_sim3("smpl_wall_similarity_refined.json")
        sim3 = load_sim3(args.smpl_wall_sim3_json) if args.smpl_wall_sim3_json is not None else None
        if sim3 is not None:
            V_wall = apply_sim3_wall(V_wall, sim3)
            J_wall = apply_sim3_wall(J_wall, sim3)

        # overlay
        canvas = bgr.copy()
        # wall verts (subsample)
        if args.draw_wall_verts and wall_V.shape[0] > 0:
            n_w = min(args.draw_wall_verts, wall_V.shape[0])
            idx_w = np.linspace(0, wall_V.shape[0]-1, n_w, dtype=int)
            uv_w, z_w = project_points_wall(wall_V[idx_w], R, t, Kt)
            draw_overlay(canvas, uv_w, (255, 0, 0), radius=1)  # blue-ish for wall

        # smpl verts (subsample)
        if args.draw_smpl_verts and V_wall.shape[0] > 0:
            n_s = min(args.draw_smpl_verts, V_wall.shape[0])
            idx_s = np.linspace(0, V_wall.shape[0]-1, n_s, dtype=int)
            uv_s, z_s = project_points_wall(V_wall[idx_s], R, t, Kt)
            draw_overlay(canvas, uv_s, (0, 255, 0), radius=1)  # green for SMPL

        # joints
        if args.draw_joints:
            uv_j, z_j = project_points_wall(J_wall, R, t, Kt)
            draw_overlay(canvas, uv_j, (0, 0, 255), radius=3)  # red for joints

        cv2.imwrite(str(out_dir / f"overlay_{Path(name).stem}.png"), canvas)
        # cv2.imwrite(str(out_dir / f"original_{Path(name).stem}.png"), bgr)

        # export combined OBJ in wall coords (every N frames to keep things small)
        if args.export_obj and (k % max(1, args.export_step) == 0):
            out_obj = exp_dir / f"combined_{Path(name).stem}.obj"
            export_combined_obj(out_obj, wall_V, wall_F, V_wall, smpl_F)

    if 'cap' in locals(): cap.release()
    print(f"[Done] overlays -> {out_dir}")
    if args.export_obj:
        print(f"[Done] combined OBJs -> {exp_dir}")
# -----------------
if __name__ == "__main__":
    main()
