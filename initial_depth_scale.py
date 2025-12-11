import torch
from pytorch3d.ops import knn_points
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

def estimate_z_scale(joints_cam, wall_vertices_cam, contact_joint_ids=[4,7,10,13], device='cuda'):
    """
    joints_cam: (T, J, 3)
    wall_vertices_cam: (Nw, 3)
    """
    joints = joints_cam[:, contact_joint_ids, :]    # (T, C, 3)
    T, C, _ = joints.shape

    q = joints.reshape(1, -1, 3).to(device)         # (1, T*C, 3)
    p = wall_vertices_cam.to(device)[None]          # (1, Nw, 3)

    d2, idx, _ = knn_points(q, p, K=1)
    p_nn = p[0, idx[0, :, 0], :]

    z_smpl = q[0, :, 2]
    z_wall = p_nn[:, 2]

    mask = torch.isfinite(z_smpl) & torch.isfinite(z_wall)
    z_smpl = z_smpl[mask]
    z_wall = z_wall[mask]

    num = torch.sum(z_smpl * z_wall)
    den = torch.sum(z_smpl * z_smpl) + 1e-8
    s_z = (num / den).item()
    return s_z


def compute_initial_depth_scale(data, body_model, verts_wall_cam, device, 
                                contact_joint_ids = [22, 23, 10, 11]   # L_Hand, R_Hand, L_Foot, R_Foot
):
    """
    Compute initial z-scale using the first training sequence.
    This wraps everything into a clean plug-and-play function.

    Arguments:
        data: the list of training sequences (data[0] is one sequence dict)
        body_model: SMPL model instance
        verts_wall_cam: (Nw,3) wall vertices in camera frame
        device: CUDA or CPU
        contact_joint_ids: which joints to consider for scale estimation

    Returns:
        init_scale (float)
    """

    print("\n[INIT] Estimating initial depth scale from first sequence...")

    # ---------------------------------------------------------
    # 1. Take first training sequence
    # ---------------------------------------------------------
    seq = data[0]
    smpl_params = seq['smpl_params']
    betas = seq['betas']

    # ---------------------------------------------------------
    # 2. Build joints_cam using SMPL
    # ---------------------------------------------------------
    with torch.no_grad():
        body_tmp = body_model.to(device)

        smpl_out = body_tmp(
            betas = betas.to(device),
            body_pose = smpl_params['body_pose'].to(device),
            global_orient = smpl_params['global_orient'].to(device),
            transl = smpl_params['transl'].to(device),
        )

        # shape: (T, J, 3)
        joints_cam = smpl_out.joints[:, :24, :].detach().cpu()

    # ---------------------------------------------------------
    # 3. Call your estimate_z_scale()
    # ---------------------------------------------------------
    init_scale = estimate_z_scale(
        joints_cam=joints_cam,
        wall_vertices_cam=verts_wall_cam.cpu(),
        contact_joint_ids=contact_joint_ids,
        device='cpu',
    )

    print(f"[INIT] Estimated depth z-scale = {init_scale:.4f}")

    return float(init_scale)

from pytorch3d.ops import knn_points

def knn_debug_distance_batch(
    verts_scaled,     # (B, V, 3)
    wall_verts,       # (W, 3)
    contact_indices,  # (C,)
    device,
    name=""
):

    with torch.no_grad():
        # verts_scaled: (B, V, 3)
        B, V, _ = verts_scaled.shape

        # -> (B, C, 3)
        contact_verts = verts_scaled[:, contact_indices, :]

        # -> (1, B*C, 3)
        p1 = contact_verts.reshape(1, -1, 3).to(device)

        # wall_verts -> (1, W, 3)
        p2 = wall_verts.to(device).unsqueeze(0)

        knn = knn_points(p1, p2, K=1)
        d = knn.dists.sqrt().view(-1)  # (B*C,)

        if d.numel() == 0:
            print(f"[{name}] no contact verts?")
            return

        mean = d.mean().item()
        median = d.median().item()
        maxv = d.max().item()

        print(f"[{name}] KNN contact-distance stats over batch:")
        print(f"  mean:   {mean:.4f} m")
        print(f"  median: {median:.4f} m")
        print(f"  max:    {maxv:.4f} m")
        print(f"  num pts: {d.numel()}")

def save_points_as_ply(points, path):
    """
    points: (N, 3) tensor on gpu/cpu
    """
    pts = points.detach().cpu().numpy()
    N = pts.shape[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")
    print(f"[save_points_as_ply] saved {N} verts to {path}")



def debug_plot_points(verts_scaled, wall_verts, epoch, out_dir="debug_vis"):
    """
    verts_scaled: (B, V, 3) tensor on cuda
    wall_verts:  (W, 3) tensor on cuda/cpu
    
    """
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # the 100th sample 
        vs = verts_scaled[0].detach().cpu().numpy()   # (V, 3)
        wv = wall_verts.detach().cpu().numpy()        # (W, 3)

        
        if vs.shape[0] > 5000:
            idx_body = np.random.choice(vs.shape[0], size=5000, replace=False)
            vs = vs[idx_body]
        if wv.shape[0] > 20000:
            idx_wall = np.random.choice(wv.shape[0], size=20000, replace=False)
            wv = wv[idx_wall]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")


        ax.scatter(
            wv[:, 0], wv[:, 1], wv[:, 2],
            s=1, alpha=0.2, label="wall"
        )
  
        ax.scatter(
            vs[:, 0], vs[:, 1], vs[:, 2],
            s=4, alpha=0.8, label="human (scaled)"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right")


        ranges = np.array([
            vs[:, 0].min(), vs[:, 0].max(),
            vs[:, 1].min(), vs[:, 1].max(),
            vs[:, 2].min(), vs[:, 2].max(),
            wv[:, 0].min(), wv[:, 0].max(),
            wv[:, 1].min(), wv[:, 1].max(),
            wv[:, 2].min(), wv[:, 2].max(),
        ]).reshape(-1, 2)
        x_min, x_max = ranges[:, 0].min(), ranges[:, 1].max()
        y_min, y_max = ranges[:, 0].min(), ranges[:, 1].max()
        z_min, z_max = ranges[:, 0].min(), ranges[:, 1].max()
        ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))


        ax.view_init(elev=20, azim=-60)

        out_path = os.path.join(out_dir, f"epoch{epoch:03d}_points.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[debug_plot_points] saved to {out_path}")
