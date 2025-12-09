
# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def render_smpl_wall_sequence(
    vertices_SMPL,      # (N, 6890, 3) - SMPL vertices sequence
    vertices_wall,      # (1, num_verts, 3) or (num_verts, 3) - Wall vertices (static)
    body_model,         # SMPL model (for faces)
    wall,              # Wall object (for faces and colors)
    image_size=(1080, 1920),  # (H, W)
    azimuth=45,        # Camera azimuth angle
    elevation=20,      # Camera elevation angle
    distance=3.0,      # Camera distance from scene
    output_dir='rendered_frames',
    save_gif=True,
    gif_path='climbing_sequence.gif',
    fps=30,
    device='cuda'
):
    """
    Render SMPL body climbing on wall for a sequence using angled view camera.
    
    Args:
        vertices_SMPL: (N, 6890, 3) - Sequence of SMPL vertices
        vertices_wall: (1, num_verts, 3) or (num_verts, 3) - Wall vertices (static)
        body_model: SMPL model
        wall: Wall object
        image_size: (H, W) tuple
        azimuth: Camera azimuth angle in degrees
        elevation: Camera elevation angle in degrees
        distance: Camera distance from scene center
        output_dir: Directory to save individual frames
        save_gif: Whether to save as GIF
        gif_path: Path for output GIF
        fps: Frames per second for GIF
        device: 'cuda' or 'cpu'
    
    Returns:
        frames: List of rendered frames as numpy arrays (H, W, 3)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    N = vertices_SMPL.shape[0]
    H, W = image_size
    
    # Handle wall vertices shape
    if vertices_wall.dim() == 3:
        vertices_wall = vertices_wall[0]  # (num_verts, 3)
    
    # Get faces
    smpl_faces = torch.tensor(body_model.faces.astype(np.int64), device=device)
    wall_faces = wall.faces
    
    # Prepare wall colors (use segmentation colors)
    wall_colors = torch.from_numpy(
        wall.color_palette[wall.vertex_regions]
    ).float().to(device)
    
    # Prepare SMPL colors (skin tone)
    skin_color = torch.tensor([0.9, 0.7, 0.6], device=device)
    
    # Get scene center for camera positioning (use wall + first frame SMPL)
    smpl_center = vertices_SMPL.mean(dim=(0, 1))  # Average across all frames and vertices
    wall_center = vertices_wall.mean(dim=0)
    scene_center = (wall_center + smpl_center) / 2
    
    print(f"Camera setup:")
    print(f"  Azimuth: {azimuth}°")
    print(f"  Elevation: {elevation}°")
    print(f"  Distance: {distance}")
    print(f"  Scene center: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
    print(f"  Image size: {W}x{H}")
    
    # Create camera using look_at_view_transform (same as Wall.render_angled_view)
    R, T = look_at_view_transform(
        dist=distance,
        elev=elevation,
        azim=azimuth,
        at=((scene_center[0].item(), scene_center[1].item(), scene_center[2].item()),),
        device=device
    )
    
    # Use same camera intrinsics as Wall class
    fx, fy = 2000, 2000  # Fixed focal length (same as Wall.render_angled_view)
    camera = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((W/2, H/2),),
        R=R,
        T=T,
        image_size=((H, W),),
        device=device,
        in_ndc=False
    )
    
    # Setup lights
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 3.0]],
        ambient_color=[[0.6, 0.6, 0.6]],
        diffuse_color=[[0.4, 0.4, 0.4]],
        specular_color=[[0.1, 0.1, 0.1]]
    )
    
    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=1000000
    )
    
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=camera,
            lights=lights
        )
    )
    
    frames = []
    
    print(f"\nRendering {N} frames...")
    for frame_idx in range(N):
        # Get SMPL vertices for this frame
        smpl_verts = vertices_SMPL[frame_idx]  # (6890, 3)
        smpl_colors = skin_color.unsqueeze(0).expand(len(smpl_verts), -1)
        
        # Combine wall and SMPL meshes
        combined_verts = torch.cat([vertices_wall, smpl_verts], dim=0)
        combined_colors = torch.cat([wall_colors, smpl_colors], dim=0)
        
        # Offset SMPL face indices
        smpl_faces_offset = smpl_faces + len(vertices_wall)
        combined_faces = torch.cat([wall_faces, smpl_faces_offset], dim=0)
        
        # Create mesh
        textures = TexturesVertex(verts_features=combined_colors.unsqueeze(0))
        mesh = Meshes(
            verts=[combined_verts],
            faces=[combined_faces],
            textures=textures
        )
        
        # Render
        rendered = renderer(mesh)
        rendered_img = rendered[..., :3].clamp(0.0, 1.0)
        
        # Convert to numpy
        frame = (rendered_img[0].cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
        
        # Save individual frame
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        imageio.imwrite(frame_path, frame)
        
        if (frame_idx + 1) % max(1, N // 10) == 0:
            print(f"  Rendered {frame_idx+1}/{N} frames")
    
    print(f"✓ Saved {N} frames to {output_dir}/")
    
    # Save as GIF
    if save_gif:
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"✓ Saved GIF: {gif_path}")
    
    return frames


def render_smpl_wall_single_frame(
    vertices_SMPL,      # (6890, 3) - Single frame SMPL vertices
    vertices_wall,      # (num_verts, 3) - Wall vertices
    body_model,         # SMPL model
    wall,              # Wall object
    image_size=(1080, 1920),  # (H, W)
    azimuth=45,
    elevation=20,
    distance=3.0,
    device='cuda'
):
    """
    Render a single frame of SMPL body on wall using angled view camera.
    
    Args:
        vertices_SMPL: (6890, 3) - SMPL vertices
        vertices_wall: (num_verts, 3) - Wall vertices
        body_model: SMPL model
        wall: Wall object
        image_size: (H, W)
        azimuth: Camera azimuth angle in degrees
        elevation: Camera elevation angle in degrees
        distance: Camera distance from scene
        device: 'cuda' or 'cpu'
    
    Returns:
        frame: (H, W, 3) rendered image as numpy array
    """
    
    H, W = image_size
    
    # Handle shapes
    if vertices_SMPL.dim() == 3:
        vertices_SMPL = vertices_SMPL[0]
    if vertices_wall.dim() == 3:
        vertices_wall = vertices_wall[0]
    
    # Get faces
    smpl_faces = torch.tensor(body_model.faces.astype(np.int64), device=device)
    wall_faces = wall.faces
    
    # Prepare colors
    wall_colors = torch.from_numpy(
        wall.color_palette[wall.vertex_regions]
    ).float().to(device)
    
    skin_color = torch.tensor([0.9, 0.7, 0.6], device=device)
    smpl_colors = skin_color.unsqueeze(0).expand(len(vertices_SMPL), -1)
    
    # Combine meshes
    combined_verts = torch.cat([vertices_wall, vertices_SMPL], dim=0)
    combined_colors = torch.cat([wall_colors, smpl_colors], dim=0)
    
    smpl_faces_offset = smpl_faces + len(vertices_wall)
    combined_faces = torch.cat([wall_faces, smpl_faces_offset], dim=0)
    
    # Get scene center for camera positioning
    scene_center = combined_verts.mean(dim=0)
    
    # Create camera using look_at_view_transform (same as Wall.render_angled_view)
    R, T = look_at_view_transform(
        dist=distance,
        elev=elevation,
        azim=azimuth,
        at=((scene_center[0].item(), scene_center[1].item(), scene_center[2].item()),),
        device=device
    )
    
    # Use same camera intrinsics as Wall class
    fx, fy = 2000, 2000  # Fixed focal length
    camera = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((W/2, H/2),),
        R=R,
        T=T,
        image_size=((H, W),),
        device=device,
        in_ndc=False
    )
    
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 3.0]],
        ambient_color=[[0.6, 0.6, 0.6]],
        diffuse_color=[[0.4, 0.4, 0.4]],
        specular_color=[[0.1, 0.1, 0.1]]
    )
    
    # Create mesh
    textures = TexturesVertex(verts_features=combined_colors.unsqueeze(0))
    mesh = Meshes(
        verts=[combined_verts],
        faces=[combined_faces],
        textures=textures
    )
    
    # Render
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=1000000
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=camera,
            lights=lights
        )
    )
    
    rendered = renderer(mesh)
    rendered_img = rendered[..., :3].clamp(0.0, 1.0)
    
    # Convert to numpy
    frame = (rendered_img[0].cpu().numpy() * 255).astype(np.uint8)
    
    return frame


# ============================================================================
# MAIN RENDERING
# ============================================================================

print(f"\nRendering full sequence...")
print(f"  Number of frames: {vertices_SMPL.shape[0]}")
print(f"  SMPL vertices per frame: {vertices_SMPL.shape[1]}")
print(f"  Wall vertices: {vertices_wall.shape}")

# Render the full sequence
rendered_frames = render_smpl_wall_sequence(
    vertices_SMPL=vertices_SMPL,
    vertices_wall=vertices_wall,
    body_model=body_model,
    wall=wall,
    image_size=(1080, 1920),  # Adjust to your image size
    azimuth=45,               # Camera horizontal angle
    elevation=20,             # Camera vertical angle
    distance=3.0,             # Camera distance from scene
    output_dir='rendered_frames',
    save_gif=True,
    gif_path='climbing_sequence.gif',
    fps=30,
    device=device
)

print("\n✓ Rendering complete!")
print(f"  Total frames: {len(rendered_frames)}")
print(f"  Output directory: rendered_frames/")
print(f"  GIF: climbing_sequence.gif")


# ============================================================================
# RENDER SINGLE FRAME (for testing/debugging)
# ============================================================================

print("\n--- Rendering single frame (frame 0) for debugging ---")
single_frame = render_smpl_wall_single_frame(
    vertices_SMPL=vertices_SMPL[0],
    vertices_wall=vertices_wall,
    body_model=body_model,
    wall=wall,
    image_size=(1080, 1920),
    azimuth=45,
    elevation=20,
    distance=3.0,
    device=device
)

imageio.imwrite('test_frame.png', single_frame)
print("✓ Saved test frame: test_frame.png")


# ============================================================================
# OPTIONAL: Render with different camera views
# ============================================================================

def render_with_angled_view(
    vertices_SMPL,
    vertices_wall,
    body_model,
    wall,
    azimuth=45,
    elevation=20,
    distance=3.0,
    image_size=(1080, 1920),
    device='cuda'
):
    """
    Render from an angled view (useful for visualization/debugging)
    """
    
    H, W = image_size
    
    if vertices_SMPL.dim() == 3:
        vertices_SMPL = vertices_SMPL[0]
    if vertices_wall.dim() == 3:
        vertices_wall = vertices_wall[0]
    
    # Get faces
    smpl_faces = torch.tensor(body_model.faces.astype(np.int64), device=device)
    wall_faces = wall.faces
    
    # Colors
    wall_colors = torch.from_numpy(
        wall.color_palette[wall.vertex_regions]
    ).float().to(device)
    
    skin_color = torch.tensor([0.9, 0.7, 0.6], device=device)
    smpl_colors = skin_color.unsqueeze(0).expand(len(vertices_SMPL), -1)
    
    # Combine
    combined_verts = torch.cat([vertices_wall, vertices_SMPL], dim=0)
    combined_colors = torch.cat([wall_colors, smpl_colors], dim=0)
    
    smpl_faces_offset = smpl_faces + len(vertices_wall)
    combined_faces = torch.cat([wall_faces, smpl_faces_offset], dim=0)
    
    # Get scene center
    scene_center = combined_verts.mean(dim=0)
    
    # Create angled camera
    R, T = look_at_view_transform(
        dist=distance,
        elev=elevation,
        azim=azimuth,
        at=((scene_center[0].item(), scene_center[1].item(), scene_center[2].item()),),
        device=device
    )
    
    camera = PerspectiveCameras(
        focal_length=((2000, 2000),),
        principal_point=((W/2, H/2),),
        R=R,
        T=T,
        image_size=((H, W),),
        device=device,
        in_ndc=False
    )
    
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 3.0]],
        ambient_color=[[0.6, 0.6, 0.6]],
        diffuse_color=[[0.4, 0.4, 0.4]],
        specular_color=[[0.1, 0.1, 0.1]]
    )
    
    # Create mesh
    textures = TexturesVertex(verts_features=combined_colors.unsqueeze(0))
    mesh = Meshes(
        verts=[combined_verts],
        faces=[combined_faces],
        textures=textures
    )
    
    # Render
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=1000000
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=camera,
            lights=lights
        )
    )
    
    rendered = renderer(mesh)
    rendered_img = rendered[..., :3].clamp(0.0, 1.0)
    
    frame = (rendered_img[0].cpu().numpy() * 255).astype(np.uint8)
    return frame


# Test angled view
print("\n--- Rendering angled view for visualization ---")
angled_frame = render_with_angled_view(
    vertices_SMPL=vertices_SMPL[0],
    vertices_wall=vertices_wall,
    body_model=body_model,
    wall=wall,
    azimuth=45,
    elevation=20,
    distance=3.0,
    device=device
)

imageio.imwrite('test_angled_view.png', angled_frame)
print("✓ Saved angled view: test_angled_view.png")