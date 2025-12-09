from wall_parameterization.Wall import Wall
import torch
import smplx
from loss.Loss import ClimbingLoss
from pdb import set_trace as st

device = 'cuda'

# From SMPL parameters and wall parameters, compute losses.


# Let's say you have sequence of SMPL parameters and wall parameters
# Initialize SMPL model
body_model = smplx.create(
        model_path="smpl_models/models",
        model_type="smpl",
        gender='male',      
        use_pca=True,      
        ext="pkl",
        batch_size=1
    ).to(device)

gvhmr_output = torch.load('/home/kunwoo/Kunwoo/GVHMR/outputs/ascendmotion/20240927JimeiYanwu_WJY_001/20240927JimeiYanwu_WJY_001_clip01/hmr4d_results.pt')

smpl_params = gvhmr_output['smpl_params_incam']
smpl_params = {k: v.to(device) for k, v in smpl_params.items()}
hands = torch.zeros(smpl_params['body_pose'].shape[0], 6, device=device)
smpl_params['body_pose'] = torch.cat([smpl_params['body_pose'], hands], dim=1)

# SMPL output
output = body_model(**smpl_params)
vertices_SMPL = output.vertices  # (N, 6890, 3)


# MESH_FILE_DIR = ''

# Initialize wall model
wall = Wall(
    mesh_file='single_view.ply',
    ref_image='wall1.png',
    segments_file='wall_mesh_segments.npy'
)


# Example 2: Forward pass with custom angles
print("\n--- Example 2: Custom angles ---")
# Rotate segment 1 by 5° in rx, segment 2 by -3° in rz
angles_custom = torch.zeros(1, wall.num_segments * 2, device=wall.device)
angles_custom[0, 0] = 5.0   # Segment 1 rx
angles_custom[0, 1] = 3.0   # Segment 1 rz
angles_custom[0, 2] = -1.0   # Segment 2 rx
angles_custom[0, 3] = -3.0  # Segment 2 rz

vertices_wall = wall.forward(angles_custom)


# NOTE ALIGN SMPL WITH WALL. THIS IS JUST ROUGH ALIGNMENT FOR NOW, IT SHOULD BE DONE WITH FEATURE MAPPING
# Step 1: Rotate SMPL 180 degrees (it's upside down)
# SMPL is in T-pose facing +Z, we want it facing the wall and right-side up
# Rotation matrix for 180 degrees around X-axis (flip upside down)
from scipy.spatial.transform import Rotation as R
rotation_matrix = R.from_euler('xz', [180, 180], degrees=True).as_matrix()
rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device=device)

# Apply rotation to all frames
vertices_SMPL_rotated = torch.matmul(vertices_SMPL, rotation_matrix.T)

# Step 2: Scale SMPL to match wall height
# Get wall height (max - min in Y direction)
wall_min_y = vertices_wall.min(dim=1)[0][:, 2].min()  # Min Y across all vertices
wall_max_y = vertices_wall.max(dim=1)[0][:, 2].max()  # Max Y across all vertices

wall_height = wall_max_y - wall_min_y

# Get SMPL height (max - min in Y direction) - averaged across all frames
smpl_min_y = vertices_SMPL_rotated.min(dim=1)[0][:, 1].min()  # Min Y across all frames
smpl_max_y = vertices_SMPL_rotated.max(dim=1)[0][:, 1].max()  # Max Y across all frames
smpl_height = smpl_max_y - smpl_min_y


# Scale factor: wall should be 3x SMPL height
target_smpl_height = wall_height / 2.0
scale_factor = target_smpl_height / smpl_height

print(f"Alignment info:")
print(f"  Wall height: {wall_height:.3f}")
print(f"  Original SMPL height: {smpl_height:.3f}")
print(f"  Target SMPL height: {target_smpl_height:.3f} (wall_height / 3)")
print(f"  Scale factor: {scale_factor:.3f}")

# Apply scaling
vertices_SMPL_scaled = vertices_SMPL_rotated * scale_factor

# Step 3: Align centers
wall_mean = vertices_wall.mean(dim=1).mean(dim=0)  # (3,)
smpl_mean = vertices_SMPL_scaled.mean(dim=1).mean(dim=0)  # (3,)
offset = wall_mean - smpl_mean

print(f"  Wall center: ({wall_mean[0]:.3f}, {wall_mean[1]:.3f}, {wall_mean[2]:.3f})")
print(f"  SMPL center (before translation): ({smpl_mean[0]:.3f}, {smpl_mean[1]:.3f}, {smpl_mean[2]:.3f})")
print(f"  Translation offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")

# Apply translation
vertices_SMPL = vertices_SMPL_scaled + offset.unsqueeze(0).unsqueeze(0)

# Verify final alignment
final_smpl_mean = vertices_SMPL.mean(dim=1).mean(dim=0)
final_smpl_height = (vertices_SMPL.max(dim=1)[0][:, 1].max() - 
                     vertices_SMPL.min(dim=1)[0][:, 1].min())
print(f"  SMPL center (after translation): ({final_smpl_mean[0]:.3f}, {final_smpl_mean[1]:.3f}, {final_smpl_mean[2]:.3f})")
print(f"  Final SMPL height: {final_smpl_height:.3f}")
print(f"  Wall/SMPL height ratio: {wall_height/final_smpl_height:.2f}x")



# Loss calculation example


# ============================================================================
# LOAD LIDAR DATA (if available)
# ============================================================================

# TODO: Load your actual LiDAR point cloud data
# For now, we'll create dummy data as placeholder
print("\n--- Loading LiDAR data ---")

# Option 1: Load from file (uncomment when you have actual data)
# lidar_points = torch.load('lidar_points.pt').to(device)

# Option 2: Create dummy data for testing
N_frames = vertices_SMPL.shape[0]
N_lidar_points = 1000  # Number of LiDAR points per frame

# Dummy LiDAR points (replace with actual data)
lidar_points = torch.randn(N_frames, N_lidar_points, 3, device=device)
# Scale and translate to be near SMPL
lidar_points = lidar_points * 0.5 + vertices_SMPL.mean(dim=1, keepdim=True)

print(f"  LiDAR points shape: {lidar_points.shape}")
print(f"  LiDAR points range: [{lidar_points.min():.3f}, {lidar_points.max():.3f}]")


# ============================================================================
# INITIALIZE LOSS FUNCTION
# ============================================================================

print("\n--- Initializing Loss Function ---")
loss_fn = ClimbingLoss(device=device)


# ============================================================================
# COMPUTE LOSSES
# ============================================================================

print("\n--- Computing Losses ---")

# Get wall vertices (remove batch dimension)
wall_verts = vertices_wall[0] if vertices_wall.dim() == 3 else vertices_wall

# Compute all losses
total_loss, losses_dict = loss_fn(
    smpl_vertices=vertices_SMPL,
    wall_vertices=wall_verts,
    lidar_points=lidar_points,
    contact_weight=1.0,      # Weight for contact loss
    depth_weight=1.0,        # Weight for depth loss
    penetration_weight=10.0, # Weight for penetration penalty (higher = stricter)
    contact_threshold=0.05,  # 5cm threshold for contact
    penetration_margin=0.01  # 1cm safety margin
)

# Print detailed metrics
loss_fn.print_metrics(losses_dict)

st()