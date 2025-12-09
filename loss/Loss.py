"""
Loss functions for climbing motion capture optimization.

Three main losses:
1. Contact Loss: Hands and feet should be on climbing holds
2. Depth Loss: SMPL body should match LiDAR depth measurements
3. Penetration Penalty Loss: Body should not penetrate through wall
"""

import torch
import torch.nn as nn
import numpy as np
from pytorch3d.ops import knn_points


class ClimbingLoss(nn.Module):
    """
    Loss functions for climbing motion capture optimization.
    """
    
    def __init__(self, device='cuda'):
        super(ClimbingLoss, self).__init__()
        self.device = device
        
        # SMPL hand and foot vertex indices (approximate)
        # These are standard SMPL vertex indices for hands and feet
        self.left_hand_indices = self._get_left_hand_indices()
        self.right_hand_indices = self._get_right_hand_indices()
        self.left_foot_indices = self._get_left_foot_indices()
        self.right_foot_indices = self._get_right_foot_indices()
        
        self.contact_indices = torch.cat([
            self.left_hand_indices,
            self.right_hand_indices,
            self.left_foot_indices,
            self.right_foot_indices
        ]).to(device)
        
        print(f"Initialized ClimbingLoss:")
        print(f"  Left hand vertices: {len(self.left_hand_indices)}")
        print(f"  Right hand vertices: {len(self.right_hand_indices)}")
        print(f"  Left foot vertices: {len(self.left_foot_indices)}")
        print(f"  Right foot vertices: {len(self.right_foot_indices)}")
        print(f"  Total contact vertices: {len(self.contact_indices)}")
    
    def _get_left_hand_indices(self):
        """Get vertex indices for left hand (palm and fingers)"""
        # SMPL left hand vertices (approximate palm and finger tips)
        # These are standard SMPL-H vertex indices
        left_hand = [
            1710, 1711, 1712, 1713, 1714,  # Palm
            1715, 1716, 1717, 1718, 1719,  # Thumb
            1720, 1721, 1722, 1723, 1724,  # Index finger
            1725, 1726, 1727, 1728, 1729,  # Middle finger
            1730, 1731, 1732, 1733, 1734,  # Ring finger
            1735, 1736, 1737, 1738, 1739,  # Pinky
        ]
        return torch.tensor(left_hand, dtype=torch.long)
    
    def _get_right_hand_indices(self):
        """Get vertex indices for right hand (palm and fingers)"""
        # SMPL right hand vertices
        right_hand = [
            5361, 5362, 5363, 5364, 5365,  # Palm
            5366, 5367, 5368, 5369, 5370,  # Thumb
            5371, 5372, 5373, 5374, 5375,  # Index finger
            5376, 5377, 5378, 5379, 5380,  # Middle finger
            5381, 5382, 5383, 5384, 5385,  # Ring finger
            5386, 5387, 5388, 5389, 5390,  # Pinky
        ]
        return torch.tensor(right_hand, dtype=torch.long)
    
    def _get_left_foot_indices(self):
        """Get vertex indices for left foot (sole and toes)"""
        # SMPL left foot vertices
        left_foot = [
            3365, 3366, 3367, 3368, 3369,  # Sole
            3370, 3371, 3372, 3373, 3374,  # Heel
            3375, 3376, 3377, 3378, 3379,  # Toes
            3380, 3381, 3382, 3383, 3384,  # Ball of foot
        ]
        return torch.tensor(left_foot, dtype=torch.long)
    
    def _get_right_foot_indices(self):
        """Get vertex indices for right foot (sole and toes)"""
        # SMPL right foot vertices
        right_foot = [
            6728, 6729, 6730, 6731, 6732,  # Sole
            6733, 6734, 6735, 6736, 6737,  # Heel
            6738, 6739, 6740, 6741, 6742,  # Toes
            6743, 6744, 6745, 6746, 6747,  # Ball of foot
        ]
        return torch.tensor(right_foot, dtype=torch.long)
    
    def contact_loss(self, smpl_vertices, wall_vertices, contact_threshold=0.05):
        """
        Contact Loss: Hand and foot vertices should be close to wall surface.
        
        Args:
            smpl_vertices: (N, 6890, 3) - SMPL vertices for N frames
            wall_vertices: (num_wall_verts, 3) - Wall vertices (static)
            contact_threshold: Distance threshold for contact (in meters)
        
        Returns:
            loss: Scalar tensor - contact loss
            metrics: Dict with detailed metrics
        """
        N = smpl_vertices.shape[0]
        
        # Handle wall vertices shape
        if wall_vertices.dim() == 3:
            wall_vertices = wall_vertices[0]
        
        # Extract contact vertices (hands and feet)
        contact_verts = smpl_vertices[:, self.contact_indices, :]  # (N, num_contact, 3)
        
        # Compute nearest neighbor distances from contact points to wall
        # Using knn_points for efficient nearest neighbor search
        wall_points = wall_vertices.unsqueeze(0).expand(N, -1, -1)  # (N, num_wall_verts, 3)
        
        # Find k=1 nearest neighbors (closest wall point to each contact point)
        knn_result = knn_points(contact_verts, wall_points, K=1)
        distances = knn_result.dists.squeeze(-1)  # (N, num_contact)
        
        # Loss: penalize contact points that are far from wall
        # Use smooth L1 loss (Huber loss) for robustness
        loss = torch.nn.functional.smooth_l1_loss(
            distances, 
            torch.zeros_like(distances),
            reduction='mean'
        )
        
        # Compute metrics
        mean_contact_dist = distances.mean().item()
        max_contact_dist = distances.max().item()
        num_in_contact = (distances < contact_threshold).sum().item()
        contact_ratio = num_in_contact / (N * len(self.contact_indices))
        
        metrics = {
            'mean_contact_distance': mean_contact_dist,
            'max_contact_distance': max_contact_dist,
            'num_vertices_in_contact': num_in_contact,
            'contact_ratio': contact_ratio
        }
        
        return loss, metrics
    
    def depth_loss(self, smpl_vertices, lidar_points, weights=None):
        """
        Depth Loss: SMPL vertices should match LiDAR point cloud measurements.
        
        Args:
            smpl_vertices: (N, 6890, 3) - SMPL vertices for N frames
            lidar_points: (N, M, 3) or (M, 3) - LiDAR point cloud
                         If (M, 3), same points used for all frames
            weights: (N, M) or (M,) - Optional confidence weights for each LiDAR point
        
        Returns:
            loss: Scalar tensor - depth loss
            metrics: Dict with detailed metrics
        """
        N = smpl_vertices.shape[0]
        
        # Handle lidar_points shape
        if lidar_points.dim() == 2:
            # Same LiDAR points for all frames
            lidar_points = lidar_points.unsqueeze(0).expand(N, -1, -1)
        
        # Manual Chamfer distance computation (more robust)
        # Forward direction: SMPL -> LiDAR
        knn_forward = knn_points(smpl_vertices, lidar_points, K=1)
        dist_forward = knn_forward.dists.squeeze(-1)  # (N, 6890)
        loss_forward = dist_forward.mean()
        
        # Backward direction: LiDAR -> SMPL
        knn_backward = knn_points(lidar_points, smpl_vertices, K=1)
        dist_backward = knn_backward.dists.squeeze(-1)  # (N, M)
        loss_backward = dist_backward.mean()
        
        # Total chamfer loss (symmetric)
        loss = loss_forward + loss_backward
        
        # Apply weights if provided
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).expand(N, -1)
            
            # Weighted backward loss (from LiDAR to SMPL)
            loss_backward_weighted = (dist_backward * weights).sum() / weights.sum()
            loss = loss_forward + loss_backward_weighted
        
        # Compute metrics
        lidar_to_smpl_dist = dist_backward.mean().item()
        smpl_to_lidar_dist = dist_forward.mean().item()
        
        metrics = {
            'chamfer_distance': loss.item(),
            'lidar_to_smpl_distance': lidar_to_smpl_dist,
            'smpl_to_lidar_distance': smpl_to_lidar_dist
        }
        
        return loss, metrics
    
    def penetration_penalty_loss(self, smpl_vertices, wall_vertices, wall_normals=None, 
                                 margin=0.01):
        """
        Penetration Penalty Loss: Body should not penetrate through wall.
        
        The wall is assumed to be a surface. We penalize any SMPL vertices that are
        "behind" the wall surface (penetrating through it).
        
        Args:
            smpl_vertices: (N, 6890, 3) - SMPL vertices for N frames
            wall_vertices: (num_wall_verts, 3) - Wall vertices (static)
            wall_normals: (num_wall_verts, 3) - Optional wall normal vectors
                         If None, normals are estimated from nearest neighbors
            margin: Safety margin distance (in meters) - penalize if closer than this
        
        Returns:
            loss: Scalar tensor - penetration penalty loss
            metrics: Dict with detailed metrics
        """
        N = smpl_vertices.shape[0]
        
        # Handle wall vertices shape
        if wall_vertices.dim() == 3:
            wall_vertices = wall_vertices[0]
        
        # Expand wall for batch processing
        wall_points = wall_vertices.unsqueeze(0).expand(N, -1, -1)  # (N, num_wall_verts, 3)
        
        # Find nearest wall point for each SMPL vertex
        knn_result = knn_points(smpl_vertices, wall_points, K=1)
        nearest_wall_idx = knn_result.idx.squeeze(-1)  # (N, 6890)
        distances = knn_result.dists.squeeze(-1)  # (N, 6890)
        
        # Get nearest wall points
        batch_indices = torch.arange(N, device=self.device).view(N, 1).expand(-1, smpl_vertices.shape[1])
        nearest_wall_points = wall_points[batch_indices, nearest_wall_idx]  # (N, 6890, 3)
        
        # Estimate wall normals if not provided
        if wall_normals is None:
            # Approximate normal as direction from wall point to SMPL point
            # (pointing away from wall)
            directions = smpl_vertices - nearest_wall_points  # (N, 6890, 3)
            wall_normals_estimated = torch.nn.functional.normalize(directions, dim=-1)
        else:
            # Use provided normals
            if wall_normals.dim() == 2:
                wall_normals = wall_normals.unsqueeze(0).expand(N, -1, -1)
            wall_normals_estimated = wall_normals[batch_indices, nearest_wall_idx]
        
        # Compute signed distance (positive = in front of wall, negative = behind/penetrating)
        # Dot product of (SMPL - wall) with wall normal
        vectors_to_smpl = smpl_vertices - nearest_wall_points
        signed_distances = (vectors_to_smpl * wall_normals_estimated).sum(dim=-1)  # (N, 6890)
        
        # Penalize vertices that are behind the wall (negative distance)
        # or within the safety margin
        penetration_depth = margin - signed_distances
        penetration_mask = penetration_depth > 0
        
        # Loss: sum of penetration depths for penetrating vertices
        if penetration_mask.any():
            loss = penetration_depth[penetration_mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Compute metrics
        num_penetrating = penetration_mask.sum().item()
        penetration_ratio = num_penetrating / (N * smpl_vertices.shape[1])
        max_penetration = penetration_depth.max().item() if penetration_mask.any() else 0.0
        mean_penetration = penetration_depth[penetration_mask].mean().item() if penetration_mask.any() else 0.0
        
        metrics = {
            'num_penetrating_vertices': num_penetrating,
            'penetration_ratio': penetration_ratio,
            'max_penetration_depth': max_penetration,
            'mean_penetration_depth': mean_penetration,
            'min_distance_to_wall': signed_distances.min().item()
        }
        
        return loss, metrics
    
    def forward(self, smpl_vertices, wall_vertices, lidar_points=None,
                contact_weight=1.0, depth_weight=1.0, penetration_weight=10.0,
                contact_threshold=0.05, penetration_margin=0.01):
        """
        Compute total loss as weighted sum of all losses.
        
        Args:
            smpl_vertices: (N, 6890, 3) - SMPL vertices
            wall_vertices: (num_wall_verts, 3) - Wall vertices
            lidar_points: (N, M, 3) or (M, 3) - LiDAR points (optional)
            contact_weight: Weight for contact loss
            depth_weight: Weight for depth loss
            penetration_weight: Weight for penetration penalty
            contact_threshold: Distance threshold for contact
            penetration_margin: Safety margin for penetration
        
        Returns:
            total_loss: Scalar tensor - weighted sum of losses
            losses_dict: Dict with individual losses and metrics
        """
        losses_dict = {}
        
        # 1. Contact Loss
        contact_loss, contact_metrics = self.contact_loss(
            smpl_vertices, wall_vertices, contact_threshold
        )
        losses_dict['contact_loss'] = contact_loss
        losses_dict['contact_metrics'] = contact_metrics
        
        # 2. Depth Loss (if LiDAR points provided)
        if lidar_points is not None:
            depth_loss, depth_metrics = self.depth_loss(smpl_vertices, lidar_points)
            losses_dict['depth_loss'] = depth_loss
            losses_dict['depth_metrics'] = depth_metrics
        else:
            depth_loss = torch.tensor(0.0, device=self.device)
            losses_dict['depth_loss'] = depth_loss
            losses_dict['depth_metrics'] = {}
        
        # 3. Penetration Penalty Loss
        penetration_loss, penetration_metrics = self.penetration_penalty_loss(
            smpl_vertices, wall_vertices, margin=penetration_margin
        )
        losses_dict['penetration_loss'] = penetration_loss
        losses_dict['penetration_metrics'] = penetration_metrics
        
        # Total weighted loss
        total_loss = (
            contact_weight * contact_loss +
            depth_weight * depth_loss +
            penetration_weight * penetration_loss
        )
        
        losses_dict['total_loss'] = total_loss
        losses_dict['weights'] = {
            'contact': contact_weight,
            'depth': depth_weight,
            'penetration': penetration_weight
        }
        
        return total_loss, losses_dict
    
    def print_metrics(self, losses_dict):
        """Print formatted loss metrics"""
        print("\n" + "="*60)
        print("LOSS METRICS")
        print("="*60)
        
        # Total loss
        print(f"\nTotal Loss: {losses_dict['total_loss'].item():.6f}")
        
        # Weights
        weights = losses_dict['weights']
        print(f"\nWeights:")
        print(f"  Contact: {weights['contact']:.2f}")
        print(f"  Depth: {weights['depth']:.2f}")
        print(f"  Penetration: {weights['penetration']:.2f}")
        
        # Individual losses
        print(f"\nIndividual Losses:")
        print(f"  Contact Loss: {losses_dict['contact_loss'].item():.6f}")
        print(f"  Depth Loss: {losses_dict['depth_loss'].item():.6f}")
        print(f"  Penetration Loss: {losses_dict['penetration_loss'].item():.6f}")
        
        # Contact metrics
        print(f"\nContact Metrics:")
        cm = losses_dict['contact_metrics']
        print(f"  Mean distance: {cm['mean_contact_distance']:.4f} m")
        print(f"  Max distance: {cm['max_contact_distance']:.4f} m")
        print(f"  Vertices in contact: {cm['num_vertices_in_contact']}")
        print(f"  Contact ratio: {cm['contact_ratio']:.2%}")
        
        # Depth metrics (if available)
        if losses_dict['depth_metrics']:
            print(f"\nDepth Metrics:")
            dm = losses_dict['depth_metrics']
            print(f"  Chamfer distance: {dm['chamfer_distance']:.4f} m")
            print(f"  LiDAR->SMPL: {dm['lidar_to_smpl_distance']:.4f} m")
            print(f"  SMPL->LiDAR: {dm['smpl_to_lidar_distance']:.4f} m")
        
        # Penetration metrics
        print(f"\nPenetration Metrics:")
        pm = losses_dict['penetration_metrics']
        print(f"  Penetrating vertices: {pm['num_penetrating_vertices']}")
        print(f"  Penetration ratio: {pm['penetration_ratio']:.2%}")
        print(f"  Max penetration: {pm['max_penetration_depth']:.4f} m")
        if pm['num_penetrating_vertices'] > 0:
            print(f"  Mean penetration: {pm['mean_penetration_depth']:.4f} m")
        print(f"  Min distance to wall: {pm['min_distance_to_wall']:.4f} m")
        
        print("="*60 + "\n")


def example_usage():
    """Example usage of ClimbingLoss"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loss module
    loss_fn = ClimbingLoss(device=device)
    
    # Create dummy data
    N = 100  # Number of frames
    smpl_vertices = torch.randn(N, 6890, 3, device=device)
    wall_vertices = torch.randn(10000, 3, device=device)
    lidar_points = torch.randn(N, 500, 3, device=device)
    
    # Compute losses
    total_loss, losses_dict = loss_fn(
        smpl_vertices=smpl_vertices,
        wall_vertices=wall_vertices,
        lidar_points=lidar_points,
        contact_weight=1.0,
        depth_weight=1.0,
        penetration_weight=10.0
    )
    
    # Print metrics
    loss_fn.print_metrics(losses_dict)
    
    # Individual loss computation
    print("\n--- Testing individual losses ---")
    
    contact_loss, contact_metrics = loss_fn.contact_loss(smpl_vertices, wall_vertices)
    print(f"Contact Loss: {contact_loss.item():.6f}")
    print(f"Contact Metrics: {contact_metrics}")
    
    depth_loss, depth_metrics = loss_fn.depth_loss(smpl_vertices, lidar_points)
    print(f"\nDepth Loss: {depth_loss.item():.6f}")
    print(f"Depth Metrics: {depth_metrics}")
    
    penetration_loss, penetration_metrics = loss_fn.penetration_penalty_loss(
        smpl_vertices, wall_vertices
    )
    print(f"\nPenetration Loss: {penetration_loss.item():.6f}")
    print(f"Penetration Metrics: {penetration_metrics}")


if __name__ == "__main__":
    example_usage()