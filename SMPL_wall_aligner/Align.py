"""
SMPL-Wall Alignment Module

Aligns SMPL mesh (from GVHMR) to Wall mesh coordinate system by matching
their projections to the same image coordinates.

The alignment is computed by:
1. Both SMPL (via GVHMR camera) and Wall (via wall camera) project correctly to image
2. We find the transformation T such that:
   wall_camera.project(T @ smpl_verts) = gvhmr_camera.project(smpl_verts)
3. This gives us the coordinate system transformation
"""

import torch
import torch.nn as nn
import numpy as np


class SMPLWallAligner(nn.Module):
    """
    Aligns SMPL vertices from GVHMR coordinate system to Wall coordinate system.
    
    The alignment consists of:
    1. Rotation (the mystery T matrix that works for wall)
    2. Scale (to match depth)
    3. Optional translation (if needed for fine alignment)
    """
    
    def __init__(self, wall, gvhmr_K, device='cuda'):
        """
        Initialize aligner.
        
        Args:
            wall: Wall object with camera and mesh
            gvhmr_K: (3, 3) or (1, 3, 3) GVHMR camera intrinsics
            device: torch device
        """
        super().__init__()
        
        self.wall = wall
        self.device = device
        
        # Store GVHMR camera matrix
        if isinstance(gvhmr_K, torch.Tensor):
            if gvhmr_K.dim() == 3:
                gvhmr_K = gvhmr_K[0]
            self.gvhmr_K = gvhmr_K.cpu().numpy()
        else:
            self.gvhmr_K = gvhmr_K
        
        # Wall camera intrinsics
        self.wall_K = wall._get_intrinsics(wall.H, wall.W)
        
        # The rotation matrix that transforms GVHMR coords to wall coords
        # This is the "mystery T matrix" that works for wall
        self.register_buffer('T_rotation', torch.tensor(
            [[-1, 0, 0], 
             [0, 0, -1], 
             [0, -1, 0]], 
            dtype=torch.float32, device=device
        ))
        
        # Scale factor (will be computed during compute_alignment)
        self.register_buffer('scale', torch.tensor(1.0, device=device))
        
        # Optional translation (usually not needed if scale is correct)
        self.register_buffer('translation', torch.zeros(3, device=device))
        
        print("✓ SMPLWallAligner initialized")
    
    def compute_alignment(self, smpl_vertices, wall_vertices, verbose=True):
        """
        Compute alignment transformation by matching projections.
        
        Args:
            smpl_vertices: (N, 6890, 3) SMPL vertices in GVHMR coordinates
            wall_vertices: (1, num_verts, 3) or (num_verts, 3) Wall vertices
            verbose: Print debug info
        
        Returns:
            scale: Computed scale factor
        """
        if verbose:
            print("\n" + "="*60)
            print("COMPUTING SMPL-WALL ALIGNMENT")
            print("="*60)
        
        # Take middle frame as reference
        mid_frame = smpl_vertices.shape[0] // 2
        smpl_verts_ref = smpl_vertices[mid_frame].cpu().numpy()  # (6890, 3)
        
        # Handle wall vertices
        if wall_vertices.dim() == 3:
            wall_verts_np = wall_vertices[0].cpu().numpy()
        else:
            wall_verts_np = wall_vertices.cpu().numpy()
        
        # 1. Project SMPL using GVHMR camera (ground truth projection)
        projected_gvhmr = (self.gvhmr_K @ smpl_verts_ref.T).T
        u_gvhmr = projected_gvhmr[:, 0] / projected_gvhmr[:, 2]
        v_gvhmr = projected_gvhmr[:, 1] / projected_gvhmr[:, 2]
        
        # Get SMPL center projection
        smpl_center_3d = smpl_verts_ref.mean(axis=0)
        p_center = self.gvhmr_K @ smpl_center_3d
        u_center = p_center[0] / p_center[2]
        v_center = p_center[1] / p_center[2]
        
        if verbose:
            print(f"\nGVHMR SMPL projection:")
            print(f"  Center projects to: ({u_center:.1f}, {v_center:.1f})")
            print(f"  u range: [{u_gvhmr.min():.1f}, {u_gvhmr.max():.1f}]")
            print(f"  v range: [{v_gvhmr.min():.1f}, {v_gvhmr.max():.1f}]")
            print(f"  SMPL depth (Z): {smpl_center_3d[2]:.3f}m")
        
        # 2. Find wall depth at SMPL location
        # Project wall vertices to image
        wall_verts_tensor = torch.from_numpy(wall_verts_np).float().to(self.device)
        points_screen = self.wall.camera.transform_points_screen(
            wall_verts_tensor.unsqueeze(0),
            image_size=((self.wall.H, self.wall.W),)
        )[0].cpu().numpy()
        
        u_wall = points_screen[:, 0]
        v_wall = points_screen[:, 1]
        z_wall = wall_verts_np[:, 2]  # Depth in wall coordinates
        
        # Find wall vertices near SMPL center
        dist_to_center = (u_wall - u_center)**2 + (v_wall - v_center)**2
        
        # Use median depth of nearest wall vertices for robustness
        n_nearest = 1000
        nearest_indices = np.argpartition(dist_to_center, n_nearest)[:n_nearest]
        reference_depth_wall = np.median(z_wall[nearest_indices])
        
        if verbose:
            print(f"\nWall mesh at SMPL location:")
            print(f"  Wall depth (Z): {reference_depth_wall:.3f}")
            print(f"  Using median of {n_nearest} nearest vertices")
        
        # 3. Compute scale
        # After rotation, SMPL will have depth ~7m (negative after transform)
        # Wall has depth ~0
        # Scale = wall_depth / smpl_depth_after_rotation
        
        # Apply rotation to SMPL to see its depth in wall coordinates
        T_rotation_np = self.T_rotation.cpu().numpy()
        smpl_rotated = smpl_verts_ref @ T_rotation_np.T
        smpl_center_rotated = smpl_rotated.mean(axis=0)
        
        # Scale factor
        scale = reference_depth_wall / smpl_center_rotated[2]
        
        if verbose:
            print(f"\nAlignment computation:")
            print(f"  SMPL center after rotation: {smpl_center_rotated}")
            print(f"  SMPL Z after rotation: {smpl_center_rotated[2]:.3f}")
            print(f"  Target wall Z: {reference_depth_wall:.3f}")
            print(f"  Scale factor: {scale:.4f}")
        
        # 4. Verify alignment by projecting transformed SMPL
        smpl_transformed = smpl_rotated * scale
        
        # Project using wall camera
        smpl_transformed_tensor = torch.from_numpy(smpl_transformed).float().to(self.device)
        points_screen_smpl = self.wall.camera.transform_points_screen(
            smpl_transformed_tensor.unsqueeze(0),
            image_size=((self.wall.H, self.wall.W),)
        )[0].cpu().numpy()
        
        u_smpl_wall = points_screen_smpl[:, 0]
        v_smpl_wall = points_screen_smpl[:, 1]
        
        # Check alignment error
        u_error = np.abs(u_smpl_wall - u_gvhmr).mean()
        v_error = np.abs(v_smpl_wall - v_gvhmr).mean()
        
        if verbose:
            print(f"\nVerification:")
            print(f"  SMPL projection error: u={u_error:.1f}px, v={v_error:.1f}px")
            print(f"  SMPL range in wall camera:")
            print(f"    u: [{u_smpl_wall.min():.1f}, {u_smpl_wall.max():.1f}]")
            print(f"    v: [{v_smpl_wall.min():.1f}, {v_smpl_wall.max():.1f}]")
            
            if u_error < 20 and v_error < 20:
                print(f"  ✓ Alignment successful (error < 20px)")
            else:
                print(f"  ⚠ Large alignment error, may need fine-tuning")
        
        # Store scale
        self.scale.data = torch.tensor(scale, dtype=torch.float32, device=self.device)
        
        return scale
    
    def apply_transform(self, smpl_vertices):
        """
        Apply alignment transformation to SMPL vertices.
        
        Args:
            smpl_vertices: (N, 6890, 3) SMPL vertices in GVHMR coordinates
        
        Returns:
            vertices_aligned: (N, 6890, 3) SMPL vertices in wall coordinates
        """
        # 1. Apply rotation
        vertices_rotated = torch.matmul(smpl_vertices, self.T_rotation.T)
        
        # 2. Apply scale
        vertices_scaled = vertices_rotated * self.scale
        
        # 3. Apply translation (if any)
        vertices_aligned = vertices_scaled + self.translation
        
        return vertices_aligned
    
    def inverse_transform(self, vertices_aligned):
        """
        Apply inverse transformation (wall coords -> GVHMR coords).
        
        Args:
            vertices_aligned: (N, 6890, 3) vertices in wall coordinates
        
        Returns:
            smpl_vertices: (N, 6890, 3) vertices in GVHMR coordinates
        """
        # Inverse operations in reverse order
        vertices = vertices_aligned - self.translation
        vertices = vertices / self.scale
        vertices = torch.matmul(vertices, self.T_rotation)  # T_rotation.T.T = T_rotation
        
        return vertices
    
    def get_transform_params(self):
        """Get current transformation parameters"""
        return {
            'T_rotation': self.T_rotation.cpu().numpy(),
            'scale': self.scale.item(),
            'translation': self.translation.cpu().numpy()
        }
    
    def set_transform_params(self, params):
        """Set transformation parameters"""
        if 'scale' in params:
            self.scale.data = torch.tensor(params['scale'], dtype=torch.float32, device=self.device)
        if 'translation' in params:
            self.translation.data = torch.tensor(params['translation'], dtype=torch.float32, device=self.device)