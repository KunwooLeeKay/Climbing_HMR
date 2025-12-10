"""
Automatic SMPL-to-Wall Alignment using Sim(3) Transformation

Based on the pipeline from:
- fit_smpl_similarity_to_wall.py: Computes similarity transform
- smpl_wall_vis_export.py: Applies transformation

This implements Umeyama's algorithm to find optimal scale, rotation, and translation
that aligns SMPL body to wall surface using nearest neighbor correspondences.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


class SMPLWallAligner:
    """
    Automatic alignment of SMPL body to wall using Sim(3) transformation.
    
    Uses Umeyama's algorithm to find:
    - Scale (s): How much to scale SMPL
    - Rotation (R): How to rotate SMPL  
    - Translation (t): How to translate SMPL
    
    Transform: X_aligned = s * (R @ X) + t
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.sim3_params = None
        
    def compute_similarity_transform(self,
                                     smpl_vertices: torch.Tensor,
                                     wall_vertices: torch.Tensor,
                                     contact_threshold: float = 0.1,
                                     num_samples: int = 2000,
                                     trim_fraction: float = 0.5,
                                     fit_mode: str = 'scale_trans'):
        """
        Compute Sim(3) transformation to align SMPL to wall.
        
        Args:
            smpl_vertices: (N, 6890, 3) or (6890, 3) - SMPL vertices
            wall_vertices: (num_verts, 3) - Wall vertices
            contact_threshold: Maximum distance for correspondence (meters)
            num_samples: Number of SMPL vertices to sample
            trim_fraction: Keep closest fraction of pairs (0-1)
            fit_mode: 'sim3' (scale+rotation+trans), 'scale_trans' (scale+trans), 
                     'scale_only' (scale only)
        
        Returns:
            sim3_params: Dict with 's', 'R', 't'
        """
        
        # Handle batch dimension
        if smpl_vertices.dim() == 3:
            # Use mean across frames for alignment
            smpl_verts = smpl_vertices.mean(dim=0)  # (6890, 3)
        else:
            smpl_verts = smpl_vertices
            
        if wall_vertices.dim() == 3:
            wall_verts = wall_vertices[0]
        else:
            wall_verts = wall_vertices
            
        # Sample SMPL vertices
        num_smpl = smpl_verts.shape[0]
        sample_size = min(num_samples, num_smpl)
        indices = torch.linspace(0, num_smpl-1, sample_size, dtype=torch.long, device=self.device)
        smpl_sampled = smpl_verts[indices]  # (sample_size, 3)
        
        # Find nearest neighbors on wall for each SMPL vertex
        # Using simple batched distance computation
        print(f"\nComputing correspondences...")
        print(f"  SMPL samples: {sample_size}")
        print(f"  Wall vertices: {wall_verts.shape[0]}")
        
        # Compute pairwise distances (batched to avoid OOM)
        batch_size = 500
        nearest_wall_points = []
        distances = []
        
        for i in range(0, sample_size, batch_size):
            end = min(i + batch_size, sample_size)
            batch_smpl = smpl_sampled[i:end]  # (batch, 3)
            
            # Compute distances to all wall vertices
            dists = torch.cdist(batch_smpl, wall_verts)  # (batch, num_wall_verts)
            min_dists, min_indices = dists.min(dim=1)  # (batch,)
            
            nearest_wall_points.append(wall_verts[min_indices])
            distances.append(min_dists)
        
        nearest_wall_points = torch.cat(nearest_wall_points, dim=0)  # (sample_size, 3)
        distances = torch.cat(distances, dim=0)  # (sample_size,)
        
        # Filter by distance threshold
        valid_mask = distances < contact_threshold
        X_smpl = smpl_sampled[valid_mask]  # Source points
        Y_wall = nearest_wall_points[valid_mask]  # Target points
        valid_dists = distances[valid_mask]
        
        print(f"  Valid correspondences: {valid_mask.sum().item()} / {sample_size}")
        print(f"  Distance range: [{valid_dists.min():.4f}, {valid_dists.max():.4f}]")
        print(f"  Median distance: {valid_dists.median():.4f}")
        
        if valid_mask.sum() < 10:
            raise RuntimeError(f"Too few correspondences ({valid_mask.sum()}). Try increasing contact_threshold.")
        
        # Trim to closest fraction
        num_keep = max(10, int(trim_fraction * len(X_smpl)))
        _, sort_indices = torch.sort(valid_dists)
        keep_indices = sort_indices[:num_keep]
        
        X_smpl = X_smpl[keep_indices]
        Y_wall = Y_wall[keep_indices]
        
        print(f"  After trimming (keep {trim_fraction:.1%}): {len(X_smpl)} pairs")
        
        # Compute Umeyama similarity transform
        with_rotation = (fit_mode == 'sim3')
        with_scaling = (fit_mode in ['sim3', 'scale_trans', 'scale_only'])
        
        if fit_mode == 'scale_only':
            with_rotation = False
            
        s, R, t = self._umeyama_similarity(
            X_smpl.cpu().numpy(), 
            Y_wall.cpu().numpy(),
            with_scaling=with_scaling,
            with_rotation=with_rotation
        )
        
        # Compute residuals
        X_transformed = s * (X_smpl.cpu().numpy() @ R.T) + t.reshape(1, 3)
        residuals = np.linalg.norm(X_transformed - Y_wall.cpu().numpy(), axis=1)
        
        print(f"\nSimilarity Transform:")
        print(f"  Mode: {fit_mode}")
        print(f"  Scale: {s:.4f}")
        print(f"  Rotation angle: {self._rotation_angle_deg(R):.2f}°")
        print(f"  Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
        print(f"  Residual median: {np.median(residuals):.4f}")
        print(f"  Residual 90th percentile: {np.percentile(residuals, 90):.4f}")
        
        self.sim3_params = {
            's': float(s),
            'R': R.astype(np.float32),
            't': t.astype(np.float32),
            'fit_mode': fit_mode
        }
        
        return self.sim3_params
    
    def _umeyama_similarity(self, X: np.ndarray, Y: np.ndarray,
                           with_scaling: bool = True,
                           with_rotation: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Umeyama's algorithm for similarity transformation.
        
        Finds s, R, t such that: Y ≈ s * (R @ X.T).T + t
        
        Args:
            X: (N, 3) source points
            Y: (N, 3) target points
            with_scaling: Include scale
            with_rotation: Include rotation
            
        Returns:
            s: scale
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        
        # Center the point clouds
        mu_X = X.mean(axis=0)
        mu_Y = Y.mean(axis=0)
        X_centered = X - mu_X
        Y_centered = Y - mu_Y
        
        # Compute rotation
        if with_rotation:
            # Covariance matrix
            C = (Y_centered.T @ X_centered) / X.shape[0]
            
            # SVD
            U, D, Vt = np.linalg.svd(C)
            
            # Handle reflection case
            S = np.eye(3)
            if np.linalg.det(U @ Vt) < 0:
                S[2, 2] = -1.0
            
            R = U @ S @ Vt
        else:
            R = np.eye(3)
        
        # Compute scale
        if with_scaling:
            var_X = (X_centered ** 2).sum() / X.shape[0]
            s = np.trace((Y_centered.T @ (X_centered @ R.T))) / (X.shape[0] * var_X)
        else:
            s = 1.0
        
        # Compute translation
        t = mu_Y - s * (R @ mu_X)
        
        return s, R, t
    
    def _rotation_angle_deg(self, R: np.ndarray) -> float:
        """Compute rotation angle in degrees from rotation matrix"""
        trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(trace)
        return float(np.degrees(angle_rad))
    
    def apply_transform(self, vertices: torch.Tensor, 
                       sim3_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply Sim(3) transformation to vertices.
        
        Args:
            vertices: (N, V, 3) or (V, 3) - vertices to transform
            sim3_params: Optional dict with 's', 'R', 't'. If None, uses self.sim3_params
            
        Returns:
            transformed_vertices: Same shape as input
        """
        if sim3_params is None:
            if self.sim3_params is None:
                raise RuntimeError("No transformation computed. Call compute_similarity_transform first.")
            sim3_params = self.sim3_params
        
        s = sim3_params['s']
        R = torch.from_numpy(sim3_params['R']).float().to(self.device)
        t = torch.from_numpy(sim3_params['t']).float().to(self.device)
        
        # Handle batch dimension
        original_shape = vertices.shape
        if vertices.dim() == 3:
            N, V, _ = vertices.shape
            vertices_flat = vertices.view(N * V, 3)
        else:
            vertices_flat = vertices
        
        # Apply: X' = s * (R @ X.T).T + t = s * (X @ R.T) + t
        transformed = s * (vertices_flat @ R.T) + t.unsqueeze(0)
        
        # Restore original shape
        if len(original_shape) == 3:
            transformed = transformed.view(N, V, 3)
        
        return transformed
    
    def save_transform(self, filepath: str):
        """Save transformation parameters to JSON"""
        if self.sim3_params is None:
            raise RuntimeError("No transformation to save")
        
        save_dict = {
            'scale': self.sim3_params['s'],
            'R': self.sim3_params['R'].tolist(),
            't': self.sim3_params['t'].tolist(),
            'fit_mode': self.sim3_params['fit_mode']
        }
        
        Path(filepath).write_text(json.dumps(save_dict, indent=2))
        print(f"✓ Saved transformation to: {filepath}")
    
    def load_transform(self, filepath: str):
        """Load transformation parameters from JSON"""
        data = json.loads(Path(filepath).read_text())
        
        self.sim3_params = {
            's': float(data['scale']),
            'R': np.array(data['R'], dtype=np.float32),
            't': np.array(data['t'], dtype=np.float32),
            'fit_mode': data.get('fit_mode', 'scale_trans')
        }
        
        print(f"✓ Loaded transformation from: {filepath}")
        print(f"  Scale: {self.sim3_params['s']:.4f}")
        print(f"  Rotation angle: {self._rotation_angle_deg(self.sim3_params['R']):.2f}°")
        
        return self.sim3_params


def align_smpl_to_wall_automatic(smpl_vertices: torch.Tensor,
                                 wall_vertices: torch.Tensor,
                                 contact_threshold: float = 0.2,
                                 num_samples: int = 2000,
                                 trim_fraction: float = 0.5,
                                 fit_mode: str = 'scale_trans',
                                 device: str = 'cuda') -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function for automatic SMPL-to-wall alignment.
    
    Args:
        smpl_vertices: (N, 6890, 3) - SMPL vertices
        wall_vertices: (num_verts, 3) - Wall vertices
        contact_threshold: Max distance for correspondences
        num_samples: Number of samples for alignment
        trim_fraction: Keep closest fraction
        fit_mode: 'sim3', 'scale_trans', or 'scale_only'
        device: 'cuda' or 'cpu'
    
    Returns:
        aligned_vertices: (N, 6890, 3) - Aligned SMPL vertices
        sim3_params: Dict with transformation parameters
    """
    aligner = SMPLWallAligner(device=device)
    
    # Compute transformation
    sim3_params = aligner.compute_similarity_transform(
        smpl_vertices,
        wall_vertices,
        contact_threshold=contact_threshold,
        num_samples=num_samples,
        trim_fraction=trim_fraction,
        fit_mode=fit_mode
    )
    
    # Apply transformation
    aligned_vertices = aligner.apply_transform(smpl_vertices, sim3_params)
    
    return aligned_vertices, sim3_params


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example data
    N_frames = 100
    smpl_vertices = torch.randn(N_frames, 6890, 3, device=device)
    wall_vertices = torch.randn(10000, 3, device=device)
    
    print("="*60)
    print("AUTOMATIC SMPL-TO-WALL ALIGNMENT")
    print("="*60)
    
    # Method 1: Using convenience function
    aligned_vertices, sim3_params = align_smpl_to_wall_automatic(
        smpl_vertices,
        wall_vertices,
        contact_threshold=0.2,
        fit_mode='scale_trans'
    )
    
    print(f"\n✓ Alignment complete!")
    print(f"  Input shape: {smpl_vertices.shape}")
    print(f"  Output shape: {aligned_vertices.shape}")
    
    # Method 2: Using class (more control)
    aligner = SMPLWallAligner(device=device)
    
    # Compute transformation
    sim3_params = aligner.compute_similarity_transform(
        smpl_vertices,
        wall_vertices,
        contact_threshold=0.2,
        num_samples=2000,
        trim_fraction=0.5,
        fit_mode='scale_trans'
    )
    
    # Save transformation
    aligner.save_transform('smpl_wall_similarity.json')
    
    # Apply to vertices
    aligned_vertices = aligner.apply_transform(smpl_vertices)
    
    # Later: Load and apply
    aligner2 = SMPLWallAligner(device=device)
    aligner2.load_transform('smpl_wall_similarity.json')
    aligned_vertices2 = aligner2.apply_transform(smpl_vertices)