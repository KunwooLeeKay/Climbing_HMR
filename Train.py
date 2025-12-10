"""
Training Pipeline for Climbing Mocap Refinement

Uses MLP networks to refine:
1. Wall deformation parameters (segment angles)
2. SMPL pose parameters (body_pose, global_orient, transl)

Loss functions:
- Contact loss: hands/feet on holds
- Penetration loss: no wall penetration
- Depth loss: match LiDAR (if available)
- Regularization: keep params close to initial values
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

from wall_parameterization.Wall import Wall
from loss.Loss import ClimbingLoss
from SMPL_wall_aligner.Align import SMPLWallAligner

from pdb import set_trace as st


# ============================================================================
# MLP Networks for Parameter Refinement
# ============================================================================

class WallRefinementMLP(nn.Module):
    """
    MLP to refine wall deformation parameters.
    
    Input: Initial wall angles (num_segments * 2,)
    Output: Refined wall angles (num_segments * 2,)
    """
    
    def __init__(self, num_segments, hidden_dims=[128, 256, 128]):
        super().__init__()
        
        input_dim = num_segments * 2  # rx, rz for each segment
        output_dim = num_segments * 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer (residual connection)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize output layer with small weights for stability
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, angles_init):
        """
        Args:
            angles_init: (batch_size, num_segments * 2) initial angles
        
        Returns:
            angles_refined: (batch_size, num_segments * 2) refined angles
        """
        delta = self.mlp(angles_init)
        # Residual connection
        angles_refined = angles_init + delta
        return angles_refined


class SMPLRefinementMLP(nn.Module):
    """
    MLP to refine SMPL parameters for a sequence.
    
    Refines per-frame parameters with temporal consistency.
    """
    
    def __init__(self, num_frames, hidden_dims=[256, 512, 256]):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Per-frame refinement
        # Input: initial SMPL params (69 for body_pose + 3 for global_orient + 3 for transl = 75)
        # Output: delta to add to initial params
        input_dim = 75  # 69 body_pose + 3 global_orient + 3 transl
        output_dim = 75
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, smpl_params_flat):
        """
        Args:
            smpl_params_flat: (batch_size, 75) flattened SMPL params
                             [body_pose (69), global_orient (3), transl (3)]
        
        Returns:
            smpl_params_refined: (batch_size, 75) refined params
        """
        delta = self.mlp(smpl_params_flat)
        # Residual connection
        smpl_params_refined = smpl_params_flat + delta
        return smpl_params_refined


# ============================================================================
# Dataset
# ============================================================================

class ClimbingMocapDataset(Dataset):
    """Dataset for climbing mocap sequences"""
    
    def __init__(self, sessions_data, device='cuda'):
        """
        Args:
            sessions_data: List of dicts with keys:
                - 'smpl_params': dict with body_pose, global_orient, transl, betas
                - 'session_name': str
                - 'gvhmr_K': (3, 3) camera intrinsics
        """
        self.sessions_data = sessions_data
        self.device = device
    
    def __len__(self):
        return len(self.sessions_data)
    
    def __getitem__(self, idx):
        return self.sessions_data[idx]


def collate_fn(batch):
    """
    Custom collate function that doesn't add batch dimension to dict values.
    Since each session is a full sequence, we don't want to batch across sessions.
    """
    # batch is a list with 1 element (batch_size=1)
    return batch[0]


# ============================================================================
# Trainer
# ============================================================================

class ClimbingMocapTrainer:
    """
    Trainer for refining wall and SMPL parameters using MLP networks.
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
        
        # Initialize MLPs
        self.wall_mlp = WallRefinementMLP(
            num_segments=wall.num_segments,
            hidden_dims=[128, 256, 128]
        ).to(device)
        
        # Will be initialized when we know sequence length
        self.smpl_mlp = None
        
        # Initial wall angles (flat wall)
        self.wall_angles_init = torch.zeros(1, wall.num_segments * 2, device=device)
        
    def prepare_smpl_params_flat(self, smpl_params):
        """
        Flatten SMPL parameters for MLP input.
        
        Args:
            smpl_params: dict with body_pose, global_orient, transl
        
        Returns:
            flat_params: (num_frames, 75) tensor
        """
        body_pose = smpl_params['body_pose']  # (N, 69)
        global_orient = smpl_params['global_orient']  # (N, 3)
        transl = smpl_params['transl']  # (N, 3)
        
        # Concatenate
        flat_params = torch.cat([body_pose, global_orient, transl], dim=1)  # (N, 75)
        return flat_params
    
    def unflatten_smpl_params(self, flat_params):
        """
        Unflatten MLP output back to SMPL parameter dict.
        
        Args:
            flat_params: (num_frames, 75) tensor
        
        Returns:
            smpl_params: dict
        """
        body_pose = flat_params[:, :69]
        global_orient = flat_params[:, 69:72]
        transl = flat_params[:, 72:75]
        
        return {
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl
        }
    
    def forward_pass(self, smpl_params_init, wall_angles_init, betas):
        """
        Complete forward pass through refinement MLPs.
        
        Args:
            smpl_params_init: dict with initial SMPL params
            wall_angles_init: (1, num_segments * 2) initial wall angles
            betas: (num_frames, 10) SMPL shape parameters
        
        Returns:
            vertices_SMPL: (num_frames, 6890, 3) refined SMPL vertices in wall coords
            vertices_wall: (num_verts, 3) refined wall vertices
            wall_angles_refined: (1, num_segments * 2) refined wall angles
            smpl_params_refined: dict with refined SMPL params
        """
        num_frames = smpl_params_init['body_pose'].shape[0]
        
        # Initialize SMPL MLP if needed
        if self.smpl_mlp is None:
            self.smpl_mlp = SMPLRefinementMLP(
                num_frames=num_frames,
                hidden_dims=[256, 512, 256]
            ).to(self.device)
        
        # 1. Refine wall angles
        wall_angles_refined = self.wall_mlp(wall_angles_init)
        vertices_wall = self.wall.forward(wall_angles_refined)
        
        # 2. Refine SMPL params per frame
        smpl_flat_init = self.prepare_smpl_params_flat(smpl_params_init)
        smpl_flat_refined = self.smpl_mlp(smpl_flat_init)
        smpl_params_refined = self.unflatten_smpl_params(smpl_flat_refined)
        smpl_params_refined['betas'] = betas
        
        # 3. Forward SMPL to get vertices (in GVHMR coordinates)
        # Process in batches to avoid OOM
        batch_size = 32
        vertices_list = []
        
        for i in range(0, num_frames, batch_size):
            end = min(i + batch_size, num_frames)
            batch_params = {
                k: v[i:end] for k, v in smpl_params_refined.items()
            }
            
            with torch.set_grad_enabled(True):
                output = self.body_model(**batch_params)
                vertices_list.append(output.vertices)
        
        vertices_SMPL = torch.cat(vertices_list, dim=0)
        
        # 4. Apply alignment transformation (GVHMR coords -> Wall coords)
        vertices_SMPL_aligned = self.aligner.apply_transform(vertices_SMPL)
        
        return vertices_SMPL_aligned, vertices_wall, wall_angles_refined, smpl_params_refined
    
    def compute_loss(self, vertices_SMPL, vertices_wall, 
                    wall_angles_refined, wall_angles_init,
                    smpl_params_refined, smpl_params_init,
                    lidar_points=None,
                    contact_weight=1.0,
                    depth_weight=1.0,
                    penetration_weight=10.0,
                    reg_wall_weight=0.1,
                    reg_smpl_weight=0.01):
        """
        Compute total loss with regularization.
        
        Args:
            vertices_SMPL: (N, 6890, 3) refined SMPL vertices
            vertices_wall: (num_verts, 3) refined wall vertices
            wall_angles_refined: (1, num_segments * 2) refined angles
            wall_angles_init: (1, num_segments * 2) initial angles
            smpl_params_refined: dict with refined SMPL params
            smpl_params_init: dict with initial SMPL params
            lidar_points: Optional LiDAR data
            contact_weight: Weight for contact loss
            depth_weight: Weight for depth loss
            penetration_weight: Weight for penetration loss
            reg_wall_weight: Weight for wall regularization
            reg_smpl_weight: Weight for SMPL regularization
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # Get wall vertices (remove batch dim)
        wall_verts = vertices_wall[0] if vertices_wall.dim() == 3 else vertices_wall
        
        # 1. Main losses (contact, depth, penetration)
        main_loss, losses_dict = self.loss_fn(
            smpl_vertices=vertices_SMPL,
            wall_vertices=wall_verts,
            lidar_points=lidar_points,
            contact_weight=contact_weight,
            depth_weight=depth_weight,
            penetration_weight=penetration_weight
        )
        
        # 2. Regularization: keep parameters close to initial values
        # Wall angle regularization
        reg_wall = torch.nn.functional.mse_loss(
            wall_angles_refined, wall_angles_init
        )
        
        # SMPL parameter regularization
        reg_smpl = 0.0
        for key in ['body_pose', 'global_orient', 'transl']:
            if key in smpl_params_refined:
                reg_smpl += torch.nn.functional.mse_loss(
                    smpl_params_refined[key],
                    smpl_params_init[key]
                )
        
        # Total loss
        total_loss = (
            main_loss +
            reg_wall_weight * reg_wall +
            reg_smpl_weight * reg_smpl
        )
        
        # Add to loss dict
        losses_dict['reg_wall'] = reg_wall
        losses_dict['reg_smpl'] = reg_smpl
        losses_dict['total_loss_with_reg'] = total_loss
        
        return total_loss, losses_dict
    
    def train_epoch(self, dataloader, optimizer, epoch, 
                   contact_weight=1.0, depth_weight=1.0, penetration_weight=10.0,
                   reg_wall_weight=0.1, reg_smpl_weight=0.01):
        """Train for one epoch"""
        
        self.wall_mlp.train()
        if self.smpl_mlp is not None:
            self.smpl_mlp.train()
        
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Get data (batch is already a dict, not a list)
            smpl_params_init = batch['smpl_params']
            betas = batch['betas']
            session_name = batch['session_name']  # Already a string, no need to unbatch
            
            # Forward pass
            vertices_SMPL, vertices_wall, wall_angles_refined, smpl_params_refined = \
                self.forward_pass(smpl_params_init, self.wall_angles_init, betas)
            
            # Compute loss
            total_loss, losses_dict = self.compute_loss(
                vertices_SMPL, vertices_wall,
                wall_angles_refined, self.wall_angles_init,
                smpl_params_refined, smpl_params_init,
                lidar_points=None,  # TODO: Add LiDAR if available
                contact_weight=contact_weight,
                depth_weight=depth_weight,
                penetration_weight=penetration_weight,
                reg_wall_weight=reg_wall_weight,
                reg_smpl_weight=reg_smpl_weight
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.wall_mlp.parameters(), max_norm=1.0)
            if self.smpl_mlp is not None:
                torch.nn.utils.clip_grad_norm_(self.smpl_mlp.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Log
            epoch_losses.append(total_loss.item())
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'contact': f"{losses_dict['contact_loss'].item():.4f}",
                'penetration': f"{losses_dict['penetration_loss'].item():.4f}"
            })
        
        return np.mean(epoch_losses)
    
    def validate(self, dataloader, contact_weight=1.0, depth_weight=1.0, penetration_weight=10.0):
        """Validate on validation set"""
        
        self.wall_mlp.eval()
        if self.smpl_mlp is not None:
            self.smpl_mlp.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                smpl_params_init = batch['smpl_params']
                betas = batch['betas']
                
                # Forward pass
                vertices_SMPL, vertices_wall, wall_angles_refined, smpl_params_refined = \
                    self.forward_pass(smpl_params_init, self.wall_angles_init, betas)
                
                # Compute loss
                total_loss, losses_dict = self.compute_loss(
                    vertices_SMPL, vertices_wall,
                    wall_angles_refined, self.wall_angles_init,
                    smpl_params_refined, smpl_params_init,
                    contact_weight=contact_weight,
                    depth_weight=depth_weight,
                    penetration_weight=penetration_weight,
                    reg_wall_weight=0.0,  # No regularization in validation
                    reg_smpl_weight=0.0
                )
                
                val_losses.append(total_loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, path, epoch, optimizer, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'wall_mlp_state_dict': self.wall_mlp.state_dict(),
            'smpl_mlp_state_dict': self.smpl_mlp.state_dict() if self.smpl_mlp else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'wall_angles_init': self.wall_angles_init,
            'aligner_params': self.aligner.get_transform_params()
        }
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.wall_mlp.load_state_dict(checkpoint['wall_mlp_state_dict'])
        if checkpoint['smpl_mlp_state_dict'] is not None:
            if self.smpl_mlp is None:
                # Initialize SMPL MLP with dummy size, will be resized on first forward
                self.smpl_mlp = SMPLRefinementMLP(num_frames=1).to(self.device)
            self.smpl_mlp.load_state_dict(checkpoint['smpl_mlp_state_dict'])
        
        self.wall_angles_init = checkpoint['wall_angles_init']
        if 'aligner_params' in checkpoint:
            self.aligner.set_transform_params(checkpoint['aligner_params'])
        
        print(f"✓ Loaded checkpoint: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Train loss: {checkpoint['train_loss']:.4f}")
        print(f"  Val loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    device = 'cuda'
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    video_path = '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train_2D'
    training_sessions_dirs = [s for s in os.listdir(video_path) if s.endswith('_images')]
    wall1_sessions = [s for s in training_sessions_dirs if s.startswith('20240927')]
    wall1_sessions = [s for s in wall1_sessions if 'WJY' in s]  # Filter
    
    # Split train/test
    training_sessions = wall1_sessions[:-1]
    testing_session = wall1_sessions[-1]
    
    print(f"Training sessions: {len(training_sessions)}")
    print(f"Testing session: {testing_session}")
    
    # Initialize SMPL model
    body_model = smplx.create(
        model_path="smpl_models",
        model_type="smpl",
        gender='male',
        use_pca=False,
        ext="pkl",
        batch_size=32
    ).to(device).eval()  # Set to eval mode
    
    # Load SMPL parameters for all sessions
    training_data = []
    for session in training_sessions:
        session_dir = session.replace("_images", "")
        smpl_path = f'/home/kunwoo/Kunwoo/GVHMR/outputs/ascendmotion_merged/{session_dir}/merged_smpl_params.pt'
        
        if not os.path.exists(smpl_path):
            print(f"Warning: {smpl_path} not found, skipping")
            continue
        
        smpl_params = torch.load(smpl_path, map_location='cpu')
        
        # Move to device
        smpl_params = {k: v.to(device) for k, v in smpl_params.items()}
        
        # Handle body_pose format
        if 'body_pose' in smpl_params:
            bp = smpl_params['body_pose']
            print(f"\n  {session_dir} body_pose shape: {bp.shape}")
            
            # Case 1: Rotation matrices (N, 23, 3, 3) -> convert to axis-angle (N, 69)
            if bp.dim() == 4 and bp.shape[-2:] == (3, 3):
                print(f"    Converting rotation matrices to axis-angle...")
                N = bp.shape[0]
                bp_flat = bp.reshape(-1, 3, 3)
                bp_aa = matrix_to_axis_angle(bp_flat)
                smpl_params['body_pose'] = bp_aa.reshape(N, -1)
                print(f"    New shape: {smpl_params['body_pose'].shape}")
            
            # Case 2: Already axis-angle but wrong shape (N, 23, 3) -> flatten to (N, 69)
            elif bp.dim() == 3 and bp.shape[1] == 23 and bp.shape[2] == 3:
                print(f"    Flattening (N, 23, 3) to (N, 69)...")
                smpl_params['body_pose'] = bp.reshape(bp.shape[0], -1)
                print(f"    New shape: {smpl_params['body_pose'].shape}")
            
            # Case 3: Axis-angle (N, 69) - already correct
            elif bp.dim() == 2 and bp.shape[1] == 69:
                print(f"    Already in correct format (N, 69)")
            
            # Case 4: Other format - need to handle
            else:
                print(f"    WARNING: Unexpected body_pose shape: {bp.shape}")
        
        # Handle global_orient format  
        if 'global_orient' in smpl_params:
            go = smpl_params['global_orient']
            print(f"  {session_dir} global_orient shape: {go.shape}")
            
            # Case 1: Rotation matrix (N, 1, 3, 3) or (N, 3, 3) -> convert to axis-angle (N, 3)
            if go.dim() == 4 and go.shape[-2:] == (3, 3):
                print(f"    Converting rotation matrices to axis-angle...")
                N = go.shape[0]
                go_flat = go.reshape(-1, 3, 3)
                go_aa = matrix_to_axis_angle(go_flat)
                smpl_params['global_orient'] = go_aa.reshape(N, -1)
                print(f"    New shape: {smpl_params['global_orient'].shape}")
            elif go.dim() == 3 and go.shape[-2:] == (3, 3):
                print(f"    Converting rotation matrices to axis-angle...")
                N = go.shape[0]
                go_aa = matrix_to_axis_angle(go)
                smpl_params['global_orient'] = go_aa.reshape(N, -1)
                print(f"    New shape: {smpl_params['global_orient'].shape}")
            
            # Case 2: Already axis-angle (N, 3) - correct
            elif go.dim() == 2 and go.shape[1] == 3:
                print(f"    Already in correct format (N, 3)")
            
            # Case 3: (N, 1, 3) -> reshape to (N, 3)
            elif go.dim() == 3 and go.shape[1] == 1 and go.shape[2] == 3:
                print(f"    Reshaping (N, 1, 3) to (N, 3)...")
                smpl_params['global_orient'] = go.squeeze(1)
                print(f"    New shape: {smpl_params['global_orient'].shape}")
            
            else:
                print(f"    WARNING: Unexpected global_orient shape: {go.shape}")
        
        training_data.append({
            'smpl_params': smpl_params,
            'betas': smpl_params.get('betas', torch.zeros(smpl_params['body_pose'].shape[0], 10, device=device)),
            'session_name': session_dir,
            'gvhmr_K': smpl_params['K_fullimg']
        })
        
        print(f"  ✓ Loaded {session_dir}: {smpl_params['body_pose'].shape[0]} frames")
    
    print(f"\n✓ Loaded {len(training_data)} training sessions")
    
    # ========================================================================
    # 2. Initialize Wall
    # ========================================================================
    
    print("\n" + "="*60)
    print("INITIALIZING WALL")
    print("="*60)
    
    wall = Wall(
        mesh_file='single_view.ply',
        ref_image='wall1.png',
        segments_file='wall_mesh_segments.npy'
    )
    
    # ========================================================================
    # 3. Initialize Alignment (use first session for initial alignment)
    # ========================================================================
    
    print("\n" + "="*60)
    print("COMPUTING INITIAL ALIGNMENT")
    print("="*60)
    
    # Compute SMPL vertices for first session
    first_smpl_params = training_data[0]['smpl_params']
    
    # Debug: Check parameter shapes
    print("\nFirst session SMPL parameter shapes:")
    for key, val in first_smpl_params.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.shape}")
    
    with torch.no_grad():
        output = body_model(**first_smpl_params)
        vertices_SMPL_first = output.vertices
    
    # Initialize wall with zero angles
    wall_angles_init = torch.zeros(1, wall.num_segments * 2, device=device)
    vertices_wall = wall.forward(wall_angles_init)
    
    # Create aligner and compute alignment
    aligner = SMPLWallAligner(
        wall=wall,
        gvhmr_K=training_data[0]['gvhmr_K'],
        device=device
    )
    
    # Compute alignment using first session
    aligner.compute_alignment(vertices_SMPL_first, vertices_wall, verbose=True)
    
    # ========================================================================
    # 4. Initialize Trainer and Loss
    # ========================================================================
    
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)
    
    loss_fn = ClimbingLoss(device=device)
    
    trainer = ClimbingMocapTrainer(
        wall=wall,
        body_model=body_model,
        aligner=aligner,
        loss_fn=loss_fn,
        device=device
    )
    
    # ========================================================================
    # 5. Create Dataloaders
    # ========================================================================
    
    train_dataset = ClimbingMocapDataset(training_data, device=device)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True,
        collate_fn=collate_fn  # Use custom collate to avoid extra batching
    )
    
    # For now, use same data for validation (you can split differently)
    val_dataset = ClimbingMocapDataset(training_data[:1], device=device)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=collate_fn  # Use custom collate to avoid extra batching
    )
    
    # ========================================================================
    # 6. Setup Optimizer
    # ========================================================================
    
    optimizer = optim.Adam([
        {'params': trainer.wall_mlp.parameters(), 'lr': 1e-4},
        # SMPL MLP will be added dynamically on first forward pass
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ========================================================================
    # 7. Training Loop
    # ========================================================================
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    num_epochs = 50
    best_val_loss = float('inf')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Add SMPL MLP to optimizer after first forward pass
        if epoch == 1 and trainer.smpl_mlp is not None:
            optimizer.add_param_group({'params': trainer.smpl_mlp.parameters(), 'lr': 1e-4})
            print("✓ Added SMPL MLP to optimizer")
        
        # Train
        train_loss = trainer.train_epoch(
            train_loader, optimizer, epoch,
            contact_weight=1.0,
            depth_weight=0.0,  # No LiDAR for now
            penetration_weight=10.0,
            reg_wall_weight=0.1,
            reg_smpl_weight=0.01
        )
        
        # Validate
        val_loss = trainer.validate(
            val_loader,
            contact_weight=1.0,
            depth_weight=0.0,
            penetration_weight=10.0
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                checkpoint_dir / 'best_model.pt',
                epoch, optimizer, train_loss, val_loss
            )
            print(f"  ✓ New best model!")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt',
                epoch, optimizer, train_loss, val_loss
            )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()