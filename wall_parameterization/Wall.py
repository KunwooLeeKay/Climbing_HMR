"""
Wall Class - SMPL-style interface for wall mesh manipulation

Similar to SMPL where:
- SMPL has pose parameters (joint angles) that deform the body mesh
- Wall has segment angles that deform the wall mesh

Usage:
    wall = Wall('single_view.ply', 'wall1.png', 'wall_mesh_segments.npy')
    
    # Forward pass
    angles = torch.tensor([[5.0, 0.0, 0.0, -3.0, 0.0]])  # [batch_size, num_segments * 2]
    vertices = wall.forward(angles)  # Returns deformed vertices
    
    # Render
    image = wall.render(vertices)
    
    # Render from angled view
    image_angled = wall.render_angled_view(vertices, azimuth=30, elevation=20)
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch3d
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from viz_utils import build_stationary_camera, get_mesh_renderer
import imageio
import cv2
import os
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

from pdb import set_trace as st


class Wall(nn.Module):
    """
    Wall mesh model with SMPL-style interface.
    
    Parameters are segment rotation angles (2 per segment: rx, rz)
    Forward pass deforms the wall mesh based on these angles.
    """
    
    def __init__(self, mesh_file, ref_image, segments_file, device=None):
        """
        Initialize Wall model.
        
        Args:
            mesh_file: Path to .ply mesh file
            ref_image: Path to reference image (for segmentation)
            segments_file: Path to .npy file with segment boundaries
            device: torch device (auto-detected if None)
        """
        super(Wall, self).__init__()
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Initializing Wall model on {self.device}...")
        
        # Load and process data
        self._load_data(mesh_file, ref_image, segments_file)
        
        # Register buffers (non-trainable but part of model state)
        self.register_buffer('verts_template', self.verts_original)
        self.register_buffer('faces', self.mesh_faces)
        self.register_buffer('vertex_regions_tensor', 
                           torch.from_numpy(self.vertex_regions).long().to(self.device))
        
        print(f"✓ Wall model initialized:")
        print(f"  - Vertices: {len(self.verts_original)}")
        print(f"  - Faces: {len(self.mesh_faces)}")
        print(f"  - Segments: {self.num_segments}")
        print(f"  - Parameters: {self.num_segments * 2} (rx, rz per segment)")
    
    def _load_data(self, mesh_file, ref_image, segments_file):
        """Load and preprocess all data"""
        
        # Load reference image
        wall_img = imageio.imread(ref_image)
        if wall_img.ndim == 3 and wall_img.shape[2] == 4:
            wall_img = wall_img[..., :3]
        self.H, self.W = wall_img.shape[:2]
        self.wall_img = wall_img
        
        # Load segments and create region map
        segments = np.load(segments_file)
        self.region_map = self._rasterize_segments_to_regions(segments, self.H, self.W)
        
        # Load mesh
        io = IO()
        mesh = io.load_mesh(mesh_file, device=self.device)
        self.verts_original = mesh.verts_packed()
        self.mesh_faces = mesh.faces_packed()
        
        # Setup camera
        T = np.array([[-1, 0, 0, 0.], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        cam_mat = self._get_intrinsics(self.H, self.W)
        self.camera, self.lights = build_stationary_camera(T, (self.H, self.W), camera_matrix=cam_mat)
        self.camera.to(self.device)
        self.lights.to(self.device)
        
        # Assign vertices to regions
        self.vertex_regions = self._assign_vertex_regions(
            self.verts_original, self.camera, self.region_map, self.H, self.W
        )
        
        self.num_segments = int(np.max(self.vertex_regions))
        
        # Generate colors for visualization
        self.color_palette = self._get_distinct_colors(self.num_segments + 1)
        self.color_palette[0] = [0.3, 0.3, 0.3]  # Background
        
        # Precompute segment centroids for faster rotation
        self.segment_centroids = {}
        for region_id in range(1, self.num_segments + 1):
            mask = self.vertex_regions == region_id
            if np.any(mask):
                region_verts = self.verts_original[mask]
                self.segment_centroids[region_id] = region_verts.mean(dim=0)
    
    def _get_intrinsics(self, H, W):
        """Get camera intrinsics"""
        f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
        cx = 0.5 * W
        cy = 0.5 * H
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    
    def _get_distinct_colors(self, n):
        """Generate distinct colors for segments"""
        cmap = plt.get_cmap('tab10')
        if n > 10:
            cmap = plt.get_cmap('tab20')
        colors = [cmap(i % cmap.N) for i in range(n)]
        np.random.seed(42)
        np.random.shuffle(colors)
        return np.array(colors)[:, :3].astype(np.float32)
    
    def _rasterize_segments_to_regions(self, segments, H, W):
        """Convert line segments to region map"""
        line_mask = np.zeros((H, W), dtype=np.uint8)
        for seg in segments:
            x1, y1, x2, y2 = seg
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)
        
        regions_binary = (255 - line_mask).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(regions_binary)
        
        dist, indices = distance_transform_edt(labels_im == 0, return_indices=True)
        filled_labels = labels_im[indices[0], indices[1]]
        
        return filled_labels
    
    def _assign_vertex_regions(self, verts, camera, region_map, H, W):
        """Assign region IDs to vertices via projection"""
        points_screen = camera.transform_points_screen(
            verts.unsqueeze(0), 
            image_size=((H, W),)
        )[0]
        
        u = points_screen[:, 0].cpu().numpy()
        v = points_screen[:, 1].cpu().numpy()
        depth = points_screen[:, 2].cpu().numpy()
        
        u_int = np.clip(np.round(u).astype(int), 0, W - 1)
        v_int = np.clip(np.round(v).astype(int), 0, H - 1)
        
        vertex_regions = region_map[v_int, u_int]
        
        invalid_mask = (depth <= 0) | (u < 0) | (u >= W) | (v < 0) | (v >= H)
        vertex_regions[invalid_mask] = 0
        
        return vertex_regions
    
    def _rotate_vertices(self, verts, rotation_angles, center):
        """
        Rotate vertices around center point.
        
        Args:
            verts: (N, 3) vertices to rotate
            rotation_angles: (rx, rz) in radians (can have gradients)
            center: (3,) rotation center
        
        Returns:
            rotated_verts: (N, 3)
        """
        rx, rz = rotation_angles
        
        # Create rotation matrices (differentiable)
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        
        # Rotation around X axis
        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=-1),
            torch.stack([zeros, torch.sin(rx), torch.cos(rx)], dim=-1)
        ], dim=-2)
        
        # Rotation around Z axis
        Rz = torch.stack([
            torch.stack([torch.cos(rz), -torch.sin(rz), zeros], dim=-1),
            torch.stack([torch.sin(rz), torch.cos(rz), zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)
        
        # Combined rotation
        R = Rz @ Rx
        
        # Rotate around center
        verts_centered = verts - center
        verts_rotated = verts_centered @ R.T
        return verts_rotated + center
    
    def forward(self, angles):
        """
        Forward pass: deform wall mesh based on segment angles.
        
        Args:
            angles: (batch_size, num_segments * 2) tensor
                   Each segment has 2 angles: [rx, rz] in degrees
                   Flattened as [seg1_rx, seg1_rz, seg2_rx, seg2_rz, ...]
        
        Returns:
            vertices: (batch_size, num_vertices, 3) deformed vertices
        """
        batch_size = angles.shape[0]
        assert angles.shape[1] == self.num_segments * 2, \
            f"Expected {self.num_segments * 2} angles, got {angles.shape[1]}"
        
        # Reshape angles: (batch_size, num_segments, 2)
        angles_reshaped = angles.view(batch_size, self.num_segments, 2)
        
        # Convert degrees to radians
        angles_rad = torch.deg2rad(angles_reshaped)
        
        # Process each batch
        vertices_batch = []
        for b in range(batch_size):
            verts = self.verts_template.clone()
            
            # Apply rotations to each segment
            for seg_idx in range(self.num_segments):
                region_id = seg_idx + 1  # Regions are 1-indexed
                
                # Get mask for this region
                mask = self.vertex_regions_tensor == region_id
                if not mask.any():
                    continue
                
                # Get vertices and rotation angles for this segment
                region_verts = verts[mask]
                rx, rz = angles_rad[b, seg_idx]
                center = self.segment_centroids[region_id]
                
                # Rotate and update
                rotated_verts = self._rotate_vertices(region_verts, (rx, rz), center)
                verts[mask] = rotated_verts
            
            vertices_batch.append(verts)
        
        # Stack into batch
        vertices = torch.stack(vertices_batch, dim=0)
        return vertices
    
    def backward_pass(self, vertices, target_vertices):
        """
        Compute gradients for optimization.
        This is handled automatically by PyTorch autograd.
        
        Args:
            vertices: (batch_size, num_vertices, 3) predicted vertices
            target_vertices: (batch_size, num_vertices, 3) target vertices
        
        Returns:
            loss: scalar tensor
        """
        # Simple L2 loss
        loss = torch.nn.functional.mse_loss(vertices, target_vertices)
        return loss
    
    def render(self, vertices, colors=None, return_overlay=False, camera=None):
        """
        Render the wall mesh.
        
        Args:
            vertices: (batch_size, num_vertices, 3) or (num_vertices, 3)
            colors: Optional vertex colors. If None, uses default segmentation colors
            return_overlay: If True, returns overlay with original image
            camera: Optional custom camera. If None, uses default camera
        
        Returns:
            rendered_img: (batch_size, H, W, 3) or (H, W, 3) if batch_size=1
        """
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            HardPhongShader
        )
        
        # Handle single vertex input
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        
        batch_size = vertices.shape[0]
        
        # Prepare colors
        if colors is None:
            vertex_colors_np = self.color_palette[self.vertex_regions]
            vertex_colors = torch.from_numpy(vertex_colors_np).float().to(self.device)
            vertex_colors = vertex_colors.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            vertex_colors = colors
        
        # Create meshes
        faces_batch = self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        textures = TexturesVertex(verts_features=vertex_colors)
        meshes = Meshes(verts=vertices, faces=faces_batch, textures=textures)
        
        # Use custom camera if provided
        cam = camera if camera is not None else self.camera
        lights = self.lights
        
        # Optimized rasterization settings for VERY dense mesh
        raster_settings = RasterizationSettings(
            image_size=(self.H, self.W),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=1000000
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cam,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cam,
                lights=lights
            )
        )
        
        rendered = renderer(meshes)
        rendered_img = rendered[..., :3].clamp(0.0, 1.0)
        
        if return_overlay:
            # Create overlay with original image
            alpha = 0.5
            wall_img_tensor = torch.from_numpy(self.wall_img).float().to(self.device) / 255.0
            overlay = alpha * wall_img_tensor + (1 - alpha) * rendered_img
            return overlay
        
        return rendered_img
    
    def render_angled_view(self, vertices, azimuth=30, elevation=45, distance=3.0, 
                          colors=None, camera=None):
        """
        Render the wall from an angled perspective view.
        
        Args:
            vertices: (num_vertices, 3) or (1, num_vertices, 3)
            azimuth: Horizontal rotation angle in degrees (0=front, positive=counterclockwise)
            elevation: Vertical angle in degrees (0=level, positive=looking down)
            distance: Camera distance from mesh center
            colors: Optional vertex colors
            camera: If provided, uses this camera instead of creating a new one
        
        Returns:
            rendered_img: (H, W, 3) rendered image
        """
        from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras
        
        # Handle single vertex input
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        
        # Get mesh center for camera to look at
        mesh_center = vertices[0].mean(dim=0)
        
        # Create camera at specified viewpoint
        R, T = look_at_view_transform(
            dist=distance,
            elev=elevation,
            azim=azimuth,
            at=((mesh_center[0].item(), mesh_center[1].item(), mesh_center[2].item()),),
            device=self.device
        )
        
        # Create camera with perspective projection
        fx, fy = 2000, 2000  # Fixed focal length
        camera_angled = PerspectiveCameras(
            focal_length=((fx, fy),),
            principal_point=((self.W/2, self.H/2),),
            R=R,
            T=T,
            image_size=((self.H, self.W),),
            in_ndc=False,
            device=self.device
        )
        
        # Render from this viewpoint
        rendered = self.render(vertices, colors=colors, camera=camera_angled)
        
        return rendered[0]
    
    def render_rotating_view(self, vertices, num_frames=24, elevation=15, distance=3.0, 
                            colors=None, output_path='rotation.gif', fps=8):
        """
        Render a rotating view of the wall mesh and save as GIF.
        
        Args:
            vertices: (num_vertices, 3) or (1, num_vertices, 3)
            num_frames: Number of frames in rotation (default 24 for faster rendering)
            elevation: Camera elevation angle in degrees
            distance: Camera distance from mesh center
            colors: Optional vertex colors
            output_path: Path to save GIF
            fps: Frames per second for GIF
        """
        from pytorch3d.renderer import look_at_view_transform
        
        # Handle single vertex input
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        
        # Get mesh center
        mesh_center = vertices[0].mean(dim=0)
        
        frames = []
        azimuths = np.linspace(0, 360, num_frames, endpoint=False)
        
        print(f"  Rendering {num_frames} frames for rotating view...")
        
        for i, azim in enumerate(azimuths):
            # Create camera at this viewpoint
            R, T = look_at_view_transform(
                dist=distance,
                elev=elevation,
                azim=azim,
                at=((mesh_center[0].item(), mesh_center[1].item(), mesh_center[2].item()),),
                device=self.device
            )
            
            # Create camera
            from pytorch3d.renderer import PerspectiveCameras
            fx, fy = 2000, 2000  # Fixed focal length for consistent view
            camera_temp = PerspectiveCameras(
                focal_length=((fx, fy),),
                principal_point=((self.W/2, self.H/2),),
                R=R,
                T=T,
                image_size=((self.H, self.W),),
                in_ndc=False,
                device=self.device
            )
            
            # Render from this viewpoint
            rendered = self.render(vertices, colors=colors, camera=camera_temp)
            frame = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
            
            if (i + 1) % 6 == 0:
                print(f"    Rendered {i+1}/{num_frames} frames")
        
        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"  ✓ Saved rotating view: {output_path}")
        
        return frames
    
    def get_zero_angles(self, batch_size=1):
        """Get zero angles (neutral pose)"""
        return torch.zeros(batch_size, self.num_segments * 2, 
                          dtype=torch.float32, device=self.device)
    
    def angles_to_dict(self, angles):
        """
        Convert flat angle tensor to dictionary format.
        
        Args:
            angles: (num_segments * 2,) tensor
        
        Returns:
            dict: {region_id: (rx, rz)}
        """
        angles_dict = {}
        for seg_idx in range(self.num_segments):
            region_id = seg_idx + 1
            rx = angles[seg_idx * 2].item()
            rz = angles[seg_idx * 2 + 1].item()
            angles_dict[region_id] = (rx, 0.0, rz)  # (rx, ry=0, rz)
        return angles_dict
    
    def dict_to_angles(self, angles_dict):
        """
        Convert dictionary format to flat angle tensor.
        
        Args:
            angles_dict: {region_id: (rx, ry, rz)} or {region_id: (rx, rz)}
        
        Returns:
            angles: (num_segments * 2,) tensor
        """
        angles = torch.zeros(self.num_segments * 2, dtype=torch.float32, device=self.device)
        for seg_idx in range(self.num_segments):
            region_id = seg_idx + 1
            if region_id in angles_dict:
                rotation = angles_dict[region_id]
                if len(rotation) == 3:
                    rx, _, rz = rotation  # Ignore ry
                else:
                    rx, rz = rotation
                angles[seg_idx * 2] = rx
                angles[seg_idx * 2 + 1] = rz
        return angles

    def save_mesh(self, vertices, output_path, colors=None):
        """
        Save the deformed mesh as a PLY file.
        
        Args:
            vertices: (num_vertices, 3) or (1, num_vertices, 3) deformed vertices
            output_path: Path to save the PLY file (e.g., 'output.ply')
            colors: Optional vertex colors (num_vertices, 3) in range [0, 1]
                   If None, uses default segmentation colors
        """
        # Handle single vertex input
        if vertices.dim() == 2:
            vertices_save = vertices
        else:
            vertices_save = vertices[0]
        
        # Prepare colors
        if colors is None:
            vertex_colors_np = self.color_palette[self.vertex_regions]
            vertex_colors = torch.from_numpy(vertex_colors_np).float().to(self.device)
        else:
            if colors.dim() == 3:
                vertex_colors = colors[0]
            else:
                vertex_colors = colors
        
        # Create mesh
        textures = TexturesVertex(verts_features=vertex_colors.unsqueeze(0))
        mesh = Meshes(
            verts=[vertices_save],
            faces=[self.faces],
            textures=textures
        )
        
        # Save using PyTorch3D IO
        io = IO()
        io.save_mesh(mesh, output_path)
        
        print(f"  ✓ Saved mesh: {output_path}")

def example_usage():
    """Example usage of Wall class"""

    os.makedirs('output_examples', exist_ok=True)
    
    # Initialize wall model
    wall = Wall(
        mesh_file='single_view.ply',
        ref_image='wall1.png',
        segments_file='wall_mesh_segments.npy'
    )
    
    # Example 1: Forward pass with zero angles
    print("\n--- Example 1: Zero angles (neutral pose) ---")
    angles_zero = wall.get_zero_angles(batch_size=1)
    vertices = wall.forward(angles_zero)
    print(f"Output vertices shape: {vertices.shape}")
    vertices_zero = vertices.clone()
    
    # Render frontal view
    rendered = wall.render(vertices)
    rendered_np = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('output_examples/wall_neutral.png', rendered_np)
    
    # Render angled view
    print("Rendering angled view (neutral pose)...")
    rendered_angled = wall.render_angled_view(vertices[0], azimuth=0, elevation=25, distance=3.0)
    rendered_angled_np = (rendered_angled.cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('output_examples/wall_neutral_angled.png', rendered_angled_np)
    print("  ✓ Saved: output_examples/wall_neutral_angled.png")
    
    # Example 2: Forward pass with custom angles
    print("\n--- Example 2: Custom angles ---")
    # Rotate segment 1 by 5° in rx, segment 2 by -3° in rz
    angles_custom = torch.zeros(1, wall.num_segments * 2, device=wall.device)
    angles_custom[0, 0] = 5.0   # Segment 1 rx
    angles_custom[0, 1] = 10.0   # Segment 1 rz
    angles_custom[0, 2] = 15.0   # Segment 2 rx
    angles_custom[0, 3] = -3.0  # Segment 2 rz
    
    vertices = wall.forward(angles_custom)
    
    # Render frontal view with overlay
    rendered = wall.render(vertices, return_overlay=True)
    rendered_np = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('output_examples/wall_custom.png', rendered_np)
    
    # Render angled view to see the deformation better
    print("Rendering angled view (custom angles)...")
    rendered_angled = wall.render_angled_view(vertices[0], azimuth=0, elevation=25, distance=3.0)
    rendered_angled_np = (rendered_angled.cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite('output_examples/wall_custom_angled.png', rendered_angled_np)
    wall.save_mesh(vertices, 'output_examples/wall_custom.ply')

    print("  ✓ Saved: output_examples/wall_custom_angled.png")
    

    print("\n✓ All examples completed! Check the output_examples/ directory.")
    print("\nKey files:")
    print("  - wall_neutral_angled.png: Angled view of neutral pose")
    print("  - wall_custom_angled.png: Angled view showing segment rotations")
    print("  - wall_neutral_top.png: Top-down view of neutral pose")
    print("  - wall_custom_top.png: Top-down view showing segment rotations")


if __name__ == "__main__":
    example_usage()