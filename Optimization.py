"""
Optimization Pipeline for Climbing Mocap Refinement

Directly optimizes:
1. Wall deformation parameters (segment angles)
2. SMPL pose parameters (body_pose, global_orient, transl)

Method:
- Initializes variables with the original Mocap/Wall data.
- Adds learnable 'deltas' to these variables.
- Minimizes Contact, Penetration, and Regularization losses using Adam.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import smplx
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from pytorch3d.transforms import matrix_to_axis_angle

# Custom imports
from wall_parameterization.Wall import Wall
from loss.Loss import ClimbingLoss
# from SMPL_wall_aligner.Align import SMPLWallAligner # REMOVED

from pdb import set_trace as st

# ============================================================================
# Optimizer Class
# ============================================================================
class ClimbingOptimizer:
    """
    Directly optimizes parameters for a specific climbing sequence.
    """
    
    def __init__(self,
                 wall: Wall,
                 body_model: smplx.SMPL,
                 loss_fn: ClimbingLoss,
                 device: str = 'cuda'):
        
        self.wall = wall
        self.body_model = body_model
        # self.aligner = aligner # REMOVED
        self.loss_fn = loss_fn
        self.device = device
        
        # Define the Wall -> Camera transformation matrix (Fixed)
        # This matches the visualization code logic
        self.R_wc = torch.tensor([
            [-1, 0, 0, 0.], 
            [0, 0, -1, 0], 
            [0, -1, 0, 0], 
            [0, 0, 0, 1]
        ], device=device).float()

    def _apply_wall_transform(self, vertices_wall_local):
        """
        Transforms Wall vertices from Local Space to Camera Space using R_wc.
        """
        # Convert to homogeneous coordinates: (N, 3) -> (N, 4)
        ones = torch.ones((vertices_wall_local.shape[0], 1), device=self.device)
        verts_homo = torch.cat([vertices_wall_local, ones], dim=1)
        
        # Apply transformation: (R @ V.T).T
        # R_wc is (4, 4), verts_homo.T is (4, N) -> Result (4, N) -> Transpose back to (N, 4)
        verts_cam_homo = (self.R_wc @ verts_homo.T).T
        
        # Return 3D coordinates
        return verts_cam_homo[:, :3]

    def _compute_batched_loss(self, 
                              vertices_SMPL, 
                              vertices_wall_cam, 
                              batch_size=32, 
                              **loss_kwargs):
        """
        Computes geometry-based losses (Contact, Penetration) in chunks.
        """
        num_frames = vertices_SMPL.shape[0]
        total_main_loss = 0.0
        
        total_loss_dict = {
            'contact_loss': 0.0,
            'penetration_loss': 0.0,
            'depth_loss': 0.0
        }

        # Loop through frames in batches
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            current_batch_size = end - i
            
            # Slice SMPL vertices for this batch
            batch_smpl_verts = vertices_SMPL[i:end]
            
            # Compute loss for this chunk
            # Note: vertices_wall_cam is static for the whole sequence (optimization step), 
            # so we pass the whole wall.
            batch_loss, batch_dict = self.loss_fn(
                smpl_vertices=batch_smpl_verts,
                wall_vertices=vertices_wall_cam, 
                **loss_kwargs
            )
            
            weight = current_batch_size / num_frames
            total_main_loss += batch_loss * weight
            
            for k, v in batch_dict.items():
                if k in total_loss_dict:
                    total_loss_dict[k] += v.detach() * weight

        return total_main_loss, total_loss_dict

    def _batch_smpl_forward(self, smpl_params, batch_size=128):
        """Helper to run SMPL forward pass in chunks."""
        num_frames = smpl_params['body_pose'].shape[0]
        vertices_list = []
        
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            batch_p = {k: v[i:end] for k, v in smpl_params.items()}
            output = self.body_model(**batch_p)
            vertices_list.append(output.vertices)
            
        return torch.cat(vertices_list, dim=0)

    def optimize_sequence(self, 
                          smpl_params_init, 
                          wall_angles_init, 
                          num_steps=500,
                          lr=1e-2):
        """
        Runs the optimization loop for a single sequence.
        """
        
        # 1. Setup Learnable Parameters (Deltas)
        wall_angles_delta = nn.Parameter(torch.zeros_like(wall_angles_init, device=self.device))
        
        betas = smpl_params_init['betas'].detach()
        body_pose_delta = nn.Parameter(torch.zeros_like(smpl_params_init['body_pose']))
        global_orient_delta = nn.Parameter(torch.zeros_like(smpl_params_init['global_orient']))
        transl_delta = nn.Parameter(torch.zeros_like(smpl_params_init['transl']))
        
        # 2. Setup Optimizer
        optimizer = optim.Adam([
            {'params': [wall_angles_delta], 'lr': lr * 0.1}, 
            {'params': [body_pose_delta, global_orient_delta, transl_delta], 'lr': lr}
        ])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        # 3. Optimization Loop
        pbar = tqdm(range(num_steps), desc="Optimizing Sequence")
        
        best_loss = float('inf')
        best_params = None
        
        for step in pbar:
            optimizer.zero_grad()
            
            # --- Construct Current State ---
            current_wall_angles = wall_angles_init + wall_angles_delta
            
            current_smpl_params = {
                'body_pose': smpl_params_init['body_pose'] + body_pose_delta,
                'global_orient': smpl_params_init['global_orient'] + global_orient_delta,
                'transl': smpl_params_init['transl'] + transl_delta,
                'betas': betas
            }
            
            # --- Forward Pass ---
            # A. Wall Forward (Local Space)
            vertices_wall_local = self.wall.forward(current_wall_angles)
            
            # B. Transform Wall to Camera Space (The key fix!)
            # We assume batch_size=1 for wall angles, so vertices_wall_local is (1, N, 3) or (N, 3)
            # Remove batch dim if exists for transform helper
            if vertices_wall_local.dim() == 3:
                v_wall_input = vertices_wall_local[0] 
            else:
                v_wall_input = vertices_wall_local
                
            vertices_wall_cam = self._apply_wall_transform(v_wall_input)
            
            # C. SMPL Forward (Already in Camera Space)
            vertices_SMPL = self._batch_smpl_forward(current_smpl_params, batch_size=64)
            
            # --- Loss Computation ---
            # Pass both in Camera Space
            main_loss, loss_dict = self._compute_batched_loss(
                vertices_SMPL=vertices_SMPL, 
                vertices_wall_cam=vertices_wall_cam,
                batch_size=32,
                lidar_points=None,
                contact_weight=1.0,
                depth_weight=0.0,
                penetration_weight=10.0
            )
            
            # Regularization Loss
            reg_pose_w = 0.05
            reg_wall_w = 1.0
            
            reg_loss = (
                reg_pose_w * torch.mean(body_pose_delta**2) +
                reg_pose_w * torch.mean(global_orient_delta**2) +
                reg_pose_w * torch.mean(transl_delta**2) +
                reg_wall_w * torch.mean(wall_angles_delta**2)
            )
            
            total_loss = main_loss + reg_loss
            
            # --- Backward ---
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # --- Tracking ---
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_params = {
                    'wall_angles': current_wall_angles.detach().clone(),
                    'smpl_params': {k: v.detach().clone() for k, v in current_smpl_params.items()},
                }
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Pen': f"{loss_dict['penetration_loss']:.4f}"
            })
            
        return best_params
    

import cv2
import glob
def render_alignment_video(vertices, K, image_dir, output_path, fps=30):
    """
    Renders the SMPL vertices projected onto the original video frames.
    """
    print(f"Rendering alignment video to {output_path}...")
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                         glob.glob(os.path.join(image_dir, "*.png")))
    
    if len(image_paths) == 0:
        print(f"Warning: No images found in {image_dir}. Skipping video.")
        return

    first_img = cv2.imread(image_paths[0])
    height, width, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(K):
        K = K.detach().cpu().numpy()
    
    if K.ndim == 3:
        K = K[0]
        
    num_frames = min(len(image_paths), vertices.shape[0])
    
    for i in tqdm(range(num_frames), desc="Rendering Video"):
        img = cv2.imread(image_paths[i])
        
        # Downsample for visualization speed
        verts_frame = vertices[i][::10] 
        
        z = verts_frame[:, 2]
        # Avoid division by zero
        z[z==0] = 1e-5
        
        x = verts_frame[:, 0] / z
        y = verts_frame[:, 1] / z
        
        u = (x * K[0, 0] + K[0, 2]).astype(int)
        v = (y * K[1, 1] + K[1, 2]).astype(int)
        
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
        u_valid = u[valid]
        v_valid = v[valid]
        
        for px, py in zip(u_valid, v_valid):
            cv2.circle(img, (px, py), 2, (255, 255, 0), -1)
            
        out.write(img)
        
    out.release()
    print("Video saved.")

# ============================================================================
# Main Script
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    wall1_sessions = ['20240927JimeiYanwu_WJY_001_images', '20240927JimeiYanwu_WJY_002_images', '20240927JimeiYanwu_WJY_003_images', '20240927JimeiYanwu_WJY_004_images', '20240927JimeiYanwu_WJY_005_images']

    target_session = wall1_sessions[0]
    session_dir = target_session.replace("_images", "")
    
    print(f"Optimizing Session: {session_dir}")
    
    smpl_path = f'ascendmotion_merged/{session_dir}/merged_smpl_params.pt'
    smpl_params = torch.load(smpl_path, map_location=device)

    # ========================================================================
    # 2. Initialize Models
    # ========================================================================
    
    body_model = smplx.create(
        model_path="smpl_models", model_type="smpl", gender='male', 
        use_pca=False, batch_size=1
    ).to(device)
    
    wall = Wall(
        mesh_file='single_view.ply',
        ref_image='wall1.png',
        segments_file='wall_mesh_segments.npy'
    )
    
    # Initial Wall Angles (Flat)
    wall_angles_init = torch.zeros(1, wall.num_segments * 2, device=device)

    # ========================================================================
    # 3. Visualization Check (No Alignment Calculation Needed!)
    # ========================================================================
    print("\n" + "="*60)
    print("VISUALIZING INITIAL STATE")
    print("="*60)

    # Since SMPL is already in Camera space, we just render it directly.
    # The wall alignment is handled internally during optimization, 
    # but for this video we just check if SMPL looks correct on the image.
    
    with torch.no_grad():
        output = body_model(**smpl_params)
        vertices_SMPL_init = output.vertices

    video_path = '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train_2D'
    image_dir_path = os.path.join(video_path, target_session) 
    
    vis_dir = Path(f'alignment_check')
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    render_alignment_video(
        vertices=vertices_SMPL_init,
        K=smpl_params['K_fullimg'],
        image_dir=image_dir_path,
        output_path=vis_dir / f'{target_session}_initial_state.mp4'
    )
    
    # ========================================================================
    # 4. Run Optimization
    # ========================================================================
    
    loss_fn = ClimbingLoss(device=device)
    
    # Initialize Optimizer (Now includes the R_wc transform internally)
    optimizer_engine = ClimbingOptimizer(
        wall=wall,
        body_model=body_model,
        loss_fn=loss_fn, # Removed aligner arg
        device=device
    )
    
    print("\nStarting Optimization...")
    
    refined_results = optimizer_engine.optimize_sequence(
        smpl_params_init=smpl_params,
        wall_angles_init=wall_angles_init,
        num_steps=300, 
        lr=1e-2
    )
    
    # ========================================================================
    # 5. Save Results
    # ========================================================================
    
    output_dir = Path(f'outputs/optimized/{session_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(refined_results['smpl_params'], output_dir / 'refined_smpl.pt')
    torch.save(refined_results['wall_angles'], output_dir / 'refined_wall.pt')
    
    print(f"\nOptimization Complete.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()