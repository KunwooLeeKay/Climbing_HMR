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

# Custom imports (Keeping your original imports)
from wall_parameterization.Wall import Wall
from loss.Loss import ClimbingLoss
from SMPL_wall_aligner.Align import SMPLWallAligner

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
                 aligner: SMPLWallAligner,
                 loss_fn: ClimbingLoss,
                 device: str = 'cuda'):
        
        self.wall = wall
        self.body_model = body_model
        self.aligner = aligner
        self.loss_fn = loss_fn
        self.device = device

    def _compute_batched_loss(self, 
                              vertices_SMPL_aligned, 
                              vertices_wall, 
                              batch_size=32, 
                              **loss_kwargs):
        """
        Computes geometry-based losses (Contact, Penetration) in chunks to save memory.
        """
        num_frames = vertices_SMPL_aligned.shape[0]
        total_main_loss = 0.0
        
        # We also want to accumulate the breakdown for logging
        total_loss_dict = {
            'contact_loss': 0.0,
            'penetration_loss': 0.0,
            'depth_loss': 0.0
        }

        # Loop through frames in batches
        for i in range(0, num_frames, batch_size):
            # 1. Slice the data
            end = min(i + batch_size, num_frames)
            current_batch_size = end - i
            
            # Slice SMPL vertices for this batch
            batch_smpl_verts = vertices_SMPL_aligned[i:end]
            
            # 2. Compute loss for this chunk
            batch_loss, batch_dict = self.loss_fn(
                smpl_vertices=batch_smpl_verts,
                wall_vertices=vertices_wall,
                **loss_kwargs
            )
            
            # 3. Accumulate (Weighted Average)
            weight = current_batch_size / num_frames
            
            # Important: Keep the computation graph for backprop!
            total_main_loss += batch_loss * weight
            
            # Accumulate metrics for logging (no grad needed usually)
            for k, v in batch_dict.items():
                if k in total_loss_dict:
                    total_loss_dict[k] += v.detach() * weight

        return total_main_loss, total_loss_dict

    def _batch_smpl_forward(self, smpl_params, batch_size=128):
        """Helper to run SMPL forward pass in chunks to save memory."""
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
        # Wall Deltas
        wall_angles_delta = nn.Parameter(
            torch.zeros_like(wall_angles_init, device=self.device)
        )
        
        # SMPL Deltas
        # Detach and clone to ensure we don't modify original data
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
            vertices_wall = self.wall.forward(current_wall_angles)
            
            # Batch the SMPL generation
            vertices_SMPL = self._batch_smpl_forward(current_smpl_params, batch_size=64)
            
            # Align
            vertices_SMPL_aligned = self.aligner.apply_transform(vertices_SMPL)
            
            # --- Loss Computation (BATCHED) ---
            main_loss, loss_dict = self._compute_batched_loss(
                vertices_SMPL_aligned=vertices_SMPL_aligned,
                vertices_wall=vertices_wall,
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
    
    
    # video_path = '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train_2D'
    # training_sessions_dirs = [s for s in os.listdir(video_path) if s.endswith('_images')]
    # wall1_sessions = [s for s in training_sessions_dirs if s.startswith('20240927') and 'WJY' in s]
    wall1_sessions = ['20240927JimeiYanwu_WJY_001_images', '20240927JimeiYanwu_WJY_002_images', '20240927JimeiYanwu_WJY_003_images', '20240927JimeiYanwu_WJY_004_images', '20240927JimeiYanwu_WJY_005_images']

    
    # Let's pick ONE session to optimize for this example
    # Optimization is usually done per-sequence
    target_session = wall1_sessions[0]
    session_dir = target_session.replace("_images", "")
    
    print(f"Optimizing Session: {session_dir}")
    
    # Load SMPL Params
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
    # 3. Setup Alignment
    # ========================================================================
    
    # Get initial vertices for alignment calculation
    with torch.no_grad():
        output = body_model(**smpl_params)
        verts_init = output.vertices
        verts_wall_init = wall.forward(wall_angles_init)
        
    aligner = SMPLWallAligner(wall=wall, gvhmr_K=smpl_params['K_fullimg'], device=device)
    aligner.compute_alignment(verts_init, verts_wall_init, verbose=True)
    
    # ========================================================================
    # 4. Run Optimization
    # ========================================================================
    
    loss_fn = ClimbingLoss(device=device)
    
    optimizer_engine = ClimbingOptimizer(
        wall=wall,
        body_model=body_model,
        aligner=aligner,
        loss_fn=loss_fn,
        device=device
    )
    
    print("\nStarting Optimization...")
    
    # Run optimization
    # Note: We pass the DATA, not a dataloader, because we optimize this specific data
    refined_results = optimizer_engine.optimize_sequence(
        smpl_params_init=smpl_params,
        wall_angles_init=wall_angles_init,
        num_steps=300, # Adjust based on need
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