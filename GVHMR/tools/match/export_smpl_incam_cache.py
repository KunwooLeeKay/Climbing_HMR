#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import torch
import numpy as np

from hmr4d.utils.smplx_utils import make_smplx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmr4d_results", required=True, help="cfg.paths.hmr4d_results (.pt saved by torch.save)")
    ap.add_argument("--outdir", required=True, help="dir to save verts_incam.npy / joints_incam.npy / meta.json")
    ap.add_argument("--frame-pattern", default="frame_%06d.jpg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {args.hmr4d_results}")
    pred = torch.load(args.hmr4d_results, map_location="cpu")

    smplx = make_smplx("supermotion").to(args.device).eval()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", map_location=args.device).to(args.device)
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", map_location=args.device).to(args.device)


    params = {k: torch.as_tensor(v).to(args.device) for k, v in pred["smpl_params_incam"].items()}
    L = params["betas"].shape[0]
    print(f"[Info] sequence length L = {L}")

    with torch.no_grad():
        smplx_out = smplx(**params)  # .vertices: list/tuple or Tensor [L, Vx, 3]
        if isinstance(smplx_out.vertices, (list, tuple)):
            verts_x = torch.stack(smplx_out.vertices, dim=0)  # [L, Vx, 3]
        else:
            verts_x = smplx_out.vertices  # [L, Vx, 3]

        # sparse matrix multiplication: project SMPL-X vertices to SMPL topology per frame
        # smplx2smpl: [Vs, Vx] sparse => (Vx,3) -> (Vs,3)
        verts_smpl = torch.stack([smplx2smpl @ verts_x[i] for i in range(L)], dim=0).contiguous()  # [L, Vs, 3]

        # joints (using SMPL neutral regressor)
        joints_smpl = torch.einsum("jv,lvk->ljk", J_regressor, verts_smpl)  # [L, J, 3]


    np.save(outdir / "verts_incam.npy", verts_smpl.cpu().numpy())
    np.save(outdir / "joints_incam.npy", joints_smpl.cpu().numpy())
    meta = {
        "L": int(L),
        "frame_pattern": args.frame_pattern, 
        "note": "Vertices/Joints are in the per-frame camera (incam) coordinate system."
    }
    json.dump(meta, open(outdir / "meta.json", "w"), indent=2)
    print(f"[Done] saved to {outdir}")

if __name__ == "__main__":
    main()
