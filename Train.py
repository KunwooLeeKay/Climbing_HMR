import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import smplx
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import pickle

# PyTorch3D Imports
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    HardPhongShader,
    PointLights,
    TexturesVertex
)

# Custom Imports
from loss.Loss import ClimbingLoss 
from wall_parameterization.Wall import Wall 

import argparse

from pdb import set_trace as st

# depth scale
from initial_depth_scale import compute_initial_depth_scale, knn_debug_distance_batch,save_points_as_ply,debug_plot_points
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Make argparse object
parser = argparse.ArgumentParser(description='Run Training')
parser.add_argument('--viz_overlay', action='store_true', help='Whether to visualize overlay videos during training')
parser.add_argument('--dataset_train_dir', type=str, help='Dataset_Train directory path', default = None)
args = parser.parse_args()


DATASET_TRAIN_DIR = args.dataset_train_dir if args.dataset_train_dir is not None else '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train'
IMG_DIR = f'{DATASET_TRAIN_DIR}_2D'
LIDAR_SCALE = 0.22

# ============================================================================
# 0. LiDAR Data Loading Function (Frame-Aligned)
# ============================================================================

def load_lidar_for_session(session_name, num_frames, device='cuda', 
                           dataset_train_dir=None, lidar_scale=0.22):
    """
    Load LiDAR point clouds for all frames of a session, ensuring frame alignment with SMPL.
    
    The key is: SMPL frame i must correspond to LiDAR frame i.
    
    Args:
        session_name: Session name (e.g., '20240927JimeiYanwu_WJY_001')
        num_frames: Number of frames in SMPL params (to ensure alignment)
        device: Device to use
        dataset_train_dir: Path to Dataset_Train directory
        lidar_scale: Scale factor for LiDAR points
    
    Returns:
        lidar_points_by_frame: List of (M_i, 3) tensors, one per frame
                               lidar_points_by_frame[i] = LiDAR points for frame i
                               Returns None if LiDAR data not available
    """
    if dataset_train_dir is None:
        dataset_train_dir = 'Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train'
    
    pkl_path = os.path.join(DATASET_TRAIN_DIR, f'filtered_{session_name}_label_visualization_IMU_GT.pkl')
    npz_path = os.path.join(DATASET_TRAIN_DIR, f'{session_name}_W2C.npz')
    

    if not os.path.exists(pkl_path) or not os.path.exists(npz_path):
        print(f"  ⚠ LiDAR data not found for {session_name}")
        return None
    
    try:
        # Load LiDAR data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        point_clouds = data['second_person']['point_clouds']  # (N_lidar_frames, 512, 3) in world coords
        
        # Load transformation matrix
        T_w2c = np.load(npz_path)['T_w2c']  # (4, 4)
        T_w2c_torch = torch.from_numpy(T_w2c).float().to(device)
        
        lidar_num_frames = point_clouds.shape[0]
        
        # Check frame count alignment
        if lidar_num_frames != num_frames:
            print(f"  ⚠ Frame count mismatch: SMPL={num_frames}, LiDAR={lidar_num_frames}")
            # Use minimum to avoid out-of-bounds
            num_frames_to_use = min(num_frames, lidar_num_frames)
        else:
            num_frames_to_use = num_frames
        
        # Transform all frames to camera coordinates (frame-by-frame)
        lidar_points_by_frame = []
        for frame_idx in range(num_frames_to_use):
            # Get LiDAR points for this specific frame
            points_world = point_clouds[frame_idx]  # (512, 3) in world coordinates
            points_world_torch = torch.from_numpy(points_world).float().to(device)
            
            # Filter valid points (remove zero-padding)
            valid_mask = points_world_torch.norm(dim=-1) > 1e-3
            points_world_valid = points_world_torch[valid_mask]
            
            if len(points_world_valid) == 0:
                # No valid points for this frame - use empty tensor
                lidar_points_by_frame.append(torch.zeros((0, 3), device=device))
                continue
            
            # Transform to camera coordinates using T_w2c
            ones = torch.ones((points_world_valid.shape[0], 1), device=device)
            points_homo = torch.cat([points_world_valid, ones], dim=1)  # (M, 4)
            points_cam_homo = (T_w2c_torch @ points_homo.T).T  # (M, 4)
            lidar_points_cam = points_cam_homo[:, :3] * lidar_scale  # (M, 3) in camera coords
            
            lidar_points_by_frame.append(lidar_points_cam)
        
        # Pad if needed to match num_frames
        if num_frames_to_use < num_frames:
            for _ in range(num_frames - num_frames_to_use):
                lidar_points_by_frame.append(torch.zeros((0, 3), device=device))
        
        print(f"  ✓ Loaded LiDAR: {num_frames_to_use} frames (aligned with SMPL frames 0-{num_frames_to_use-1})")
        return lidar_points_by_frame  # List where lidar_points_by_frame[i] = LiDAR for frame i
        
    except Exception as e:
        print(f"  ⚠ Error loading LiDAR for {session_name}: {e}")
        return None

# ============================================================================
# 1. Visualization Function (Updated for Training Loop)
# ============================================================================

def verify_alignment_video(session_data, wall_obj, verts_wall_cam, device, output_path="verification_alignment.mp4", refined_verts=None):
    """
    Generates a video verifying the alignment between SMPL and Wall.
    Args:
        refined_verts: (Optional) Tensor of shape (N, 6890, 3). If provided, these vertices are rendered 
                       instead of generating them from the initial SMPL params.
    """
    print(f"\nGenerating video: {output_path}...")
    
    # Unpack session data
    smpl_params = session_data['smpl_params']
    session_name = session_data['session_name']
    
    # 1. Setup Image Paths
    img_dir = f'{IMG_DIR}/{session_name}_images'
    
    # Check if image directory exists, if not try alternative paths
    if not os.path.exists(img_dir):
        # Try without _images suffix
        img_dir_alt = f'{IMG_DIR}/{session_name}'
        if os.path.exists(img_dir_alt):
            img_dir = img_dir_alt
        else:
            print(f"Warning: Image directory not found at {img_dir}")
            print(f"Skipping video generation for {session_name}")
            return
    
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if len(images) == 0:
        print(f"Warning: No images found in {img_dir}")
        print(f"Skipping video generation for {session_name}")
        return
    
    # Load first image for resolution
    first_img = cv2.imread(os.path.join(img_dir, images[0]))
    H_vid, W_vid, _ = first_img.shape
    
    # 2. Setup Wall Reference for Texture & K Scaling
    wall_ref_img = cv2.imread('wall1.png')
    if wall_ref_img is None:
        # print("Warning: wall1.png not found. Using solid color for wall.")
        H_ref, W_ref = H_vid, W_vid
        use_texture = False
    else:
        H_ref, W_ref, _ = wall_ref_img.shape
        use_texture = True

    # Calculate Scaling Factors
    scale_x = W_vid / W_ref
    scale_y = H_vid / H_ref
    
    # 3. Setup Cameras
    K_body = session_data['gvhmr_K'][0].cpu().numpy()
    
    K_wall_orig = np.array([
        [2.69417743e+03, 0.00000000e+00, 1.40250000e+03],
        [0.00000000e+00, 2.69417743e+03, 8.77500000e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    K_wall_scaled = K_wall_orig.copy()
    K_wall_scaled[0, 0] *= scale_x; K_wall_scaled[1, 1] *= scale_y
    K_wall_scaled[0, 2] *= scale_x; K_wall_scaled[1, 2] *= scale_y
    
    def create_cam(K, H, W):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R_corr = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], device=device).unsqueeze(0)
        return PerspectiveCameras(focal_length=((fx, fy),), principal_point=((cx, cy),), 
                                  image_size=((H, W),), R=R_corr, T=torch.zeros((1, 3), device=device), 
                                  in_ndc=False, device=device)

    cam_body = create_cam(K_body, H_vid, W_vid)
    cam_wall = create_cam(K_wall_scaled, H_vid, W_vid)
    
    # 4. Texture for Wall
    if use_texture:
        ref_tensor = torch.from_numpy(wall_obj.wall_img).float().to(device) / 255.0
        H_r, W_r = ref_tensor.shape[:2]
        with torch.no_grad():
            pts = wall_obj.camera.transform_points_screen(wall_obj.verts_template.unsqueeze(0), image_size=((H_r, W_r),))[0]
            u, v = pts[:, 0], pts[:, 1]
            u_n = 2.0 * (u / (W_r - 1)) - 1.0; v_n = 2.0 * (v / (H_r - 1)) - 1.0
            grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(0).unsqueeze(0)
            img_batch = ref_tensor.permute(2, 0, 1).unsqueeze(0)
            sampled = F.grid_sample(img_batch, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
            verts_rgb = sampled.squeeze().permute(1, 0)
        textures_wall = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
    else:
        verts_rgb = torch.ones_like(verts_wall_cam) * torch.tensor([0.2, 0.8, 0.2], device=device)
        textures_wall = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))

    # 5. Create Wall Mesh
    mesh_wall = Meshes(verts=[verts_wall_cam], faces=[wall_obj.faces], textures=textures_wall)
    
    # 6. Renderer
    raster_settings = RasterizationSettings(image_size=(H_vid, W_vid), blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    
    def get_renderer(cam):
        return MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster_settings),
                            shader=HardPhongShader(device=device, cameras=cam, lights=lights))

    rend_wall = get_renderer(cam_wall)
    rend_body = get_renderer(cam_body)
    
    # Pre-render wall
    out_wall = rend_wall(mesh_wall)
    img_wall_rgb = out_wall[0, ..., :3]
    mask_wall = out_wall[0, ..., 3] > 0
    
    # 7. Render Loop
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W_vid, H_vid))
    
    # Decide which vertices to use
    if refined_verts is not None:
        verts_smpl = refined_verts
        N_vis = min(100, len(verts_smpl))
    else:
        body_model = smplx.create(model_path="smpl_models", model_type="smpl", gender='male', ext="pkl").to(device).eval()
        with torch.no_grad():
            N_vis = min(100, len(images))
            params_vis = {k: v[:N_vis] for k, v in smpl_params.items() if k in ['body_pose', 'global_orient', 'transl', 'betas']}
            verts_smpl = body_model(**params_vis).vertices
        
    for i in tqdm(range(N_vis), desc="Rendering Video", leave=False):
        if i >= len(images):
            print(f"Warning: Frame {i} exceeds available images ({len(images)})")
            break
            
        img_path = os.path.join(img_dir, images[i])
        bg = cv2.imread(img_path)
        if bg is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        
        # Render Body
        v_frame = verts_smpl[i]
        
        color_tensor = (torch.ones_like(v_frame) * torch.tensor([0.8, 0.2, 0.2], device=device)).unsqueeze(0)
        tex_body = TexturesVertex(verts_features=color_tensor)
        
        # Need dummy face tensor if body model wasn't created in this scope
        # Use wall_obj.faces or load a dummy model to get faces if necessary
        # Assuming body_model or similar logic is available to get faces. 
        # Since faces are constant, we can borrow them from the trainer's model or create a temp one.
        if 'body_faces_tensor' not in locals():
             temp_body = smplx.create(model_path="smpl_models", model_type="smpl", gender='male', ext="pkl").to(device)
             body_faces_tensor = temp_body.faces_tensor
             
        mesh_body = Meshes(
            verts=[v_frame], 
            faces=[body_faces_tensor], 
            textures=tex_body
        )
        
        out_body = rend_body(mesh_body)
        img_body = out_body[0, ..., :3]
        mask_body = out_body[0, ..., 3] > 0
        
        # Composite
        bg_t = torch.flip(torch.from_numpy(bg).float().to(device)/255.0, dims=[-1])
        final = bg_t.clone()
        final[mask_wall] = 0.7 * img_wall_rgb[mask_wall] + 0.3 * final[mask_wall]
        final[mask_body] = img_body[mask_body]
        
        final_bgr = torch.flip(final, dims=[-1]).cpu().numpy() * 255
        writer.write(final_bgr.astype(np.uint8))
        
    writer.release()
    print(f"✓ Video saved to {output_path}")


def save_meshes_for_sequence(refined_verts, wall_verts, wall_faces, body_faces, output_dir, session_name, device):
    """
    Save refined body meshes and wall mesh as OBJ files for visualization.
    
    Args:
        refined_verts: Tensor of shape (N, 6890, 3) - refined body vertices
        wall_verts: Tensor of shape (W, 3) - wall vertices
        wall_faces: Tensor of shape (F_wall, 3) - wall faces
        body_faces: Tensor of shape (F_body, 3) - body faces
        output_dir: Directory to save meshes
        session_name: Name of the session
        device: Device
    """
    print(f"\nSaving meshes for {session_name}...")
    
    # Create output directory
    mesh_dir = Path(output_dir) / session_name / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    # Save wall mesh (only once)
    wall_mesh_path = mesh_dir / "wall.obj"
    save_obj_mesh(wall_verts.cpu().numpy(), wall_faces.cpu().numpy(), wall_mesh_path)
    print(f"  ✓ Saved wall mesh: {wall_mesh_path}")
    
    # Save body meshes (sample every 10 frames to avoid too many files)
    num_frames = len(refined_verts)
    sample_interval = max(1, num_frames // 50)  # Save ~50 frames max
    
    for i in tqdm(range(0, num_frames, sample_interval), desc="Saving body meshes"):
        body_mesh_path = mesh_dir / f"body_frame_{i:04d}.obj"
        save_obj_mesh(refined_verts[i].cpu().numpy(), body_faces.cpu().numpy(), body_mesh_path)
    
    print(f"  ✓ Saved {len(range(0, num_frames, sample_interval))} body meshes")
    
    # Also save a combined mesh for the first frame
    combined_path = mesh_dir / "frame_0000_combined.obj"
    save_combined_obj(refined_verts[0].cpu().numpy(), body_faces.cpu().numpy(),
                      wall_verts.cpu().numpy(), wall_faces.cpu().numpy(), combined_path)
    print(f"  ✓ Saved combined mesh: {combined_path}")


def save_obj_mesh(vertices, faces, filepath):
    """Save a mesh to OBJ file."""
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_combined_obj(body_verts, body_faces, wall_verts, wall_faces, filepath):
    """Save body and wall as a single OBJ file."""
    with open(filepath, 'w') as f:
        # Write body vertices
        f.write("# Body vertices\n")
        for v in body_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write wall vertices
        f.write("# Wall vertices\n")
        for v in wall_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write body faces
        f.write("# Body faces\n")
        for face in body_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Write wall faces (offset by number of body vertices)
        f.write("# Wall faces\n")
        offset = len(body_verts)
        for face in wall_faces:
            f.write(f"f {face[0]+1+offset} {face[1]+1+offset} {face[2]+1+offset}\n")


# ============================================================================
# 2. Training Classes
# ============================================================================

class SMPLRefinementMLP(nn.Module):
    def __init__(self, num_frames, hidden_dims=[256, 512, 256]):
        super().__init__()
        input_dim = 75 
        output_dim = 75
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.mlp[-1].weight, std=0.001); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x): return x + self.mlp(x)

class ClimbingMocapDataset(Dataset):
    def __init__(self, data, device='cuda'): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(b): return b[0]

class ClimbingMocapTrainer:
    def __init__(self, wall_verts, wall_faces, body_model, loss_fn, device='cuda'):
        self.wall_verts = wall_verts.to(device)
        self.wall_faces = wall_faces.to(device)
        self.body_model = body_model
        self.loss_fn = loss_fn
        self.device = device
        self.smpl_mlp = None
        self.depth_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.contact_indices = self.loss_fn.contact_indices.to(self.device)
        
    def forward_pass(self, params_init, betas):
        if self.smpl_mlp is None:
            self.smpl_mlp = SMPLRefinementMLP(params_init['body_pose'].shape[0]).to(self.device)
        
        flat = torch.cat([params_init['body_pose'], params_init['global_orient'], params_init['transl']], 1)
        refined = self.smpl_mlp(flat)
        params_ref = {'body_pose': refined[:,:69], 'global_orient': refined[:,69:72], 'transl': refined[:,72:75], 'betas': betas}
        
        # Batched SMPL Forward
        verts_list = []
        for i in range(0, len(flat), 32):
            p = {k: v[i:i+32] for k, v in params_ref.items()}
            verts_list.append(self.body_model(**p).vertices)
        return torch.cat(verts_list, 0), params_ref
    
    def train_epoch(self, loader, opt, epoch, **kwargs):
        if self.smpl_mlp: self.smpl_mlp.train()
        losses = []
        tot_con_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            opt.zero_grad()
            if not self.smpl_mlp: 
                self.forward_pass(batch['smpl_params'], batch['betas'])
                opt.add_param_group({'params': self.smpl_mlp.parameters(), 'lr': 1e-4})
                self.smpl_mlp.train()
            
            verts, params_ref = self.forward_pass(batch['smpl_params'], batch['betas'])
            # depth scale
            scale_vec = torch.stack([
                torch.ones((), device=verts.device), 
                torch.ones((), device=verts.device), 
                self.depth_scale 
            ])
            verts_scaled = verts * scale_vec.view(1, 1, 3)
            if batch_idx == 0 and epoch % 1 == 0:
                knn_debug_distance_batch(
                    verts_scaled,          # (B,V,3)
                    self.wall_verts,       # (W,3)
                    self.contact_indices,  # (C,)
                    self.device,
                    name=f"epoch{epoch}_batch0"
                )
                print("self.depth_scale", self.depth_scale.detach())
                debug_plot_points(verts_scaled, self.wall_verts, epoch)
                body_pts = verts_scaled[100]          # (V,3)
                wall_pts = self.wall_verts          # (W,3)
                save_points_as_ply(body_pts, "debug_vis/body_epoch0.ply")
                save_points_as_ply(wall_pts, "debug_vis/wall.ply")

            tot_pen = 0; tot_con = 0; tot_loss = 0
            chunk = 64
            for i in range(0, len(verts), chunk):
                v_chunk = verts_scaled[i:i+chunk]

                l_main, l_dict = self.loss_fn(v_chunk, self.wall_verts, None, **kwargs)
                weight = len(v_chunk)/len(verts)
                tot_pen += l_dict['penetration_loss'] * weight
                tot_con += l_dict['contact_loss'] * weight
            
            reg = sum(F.mse_loss(params_ref[k], batch['smpl_params'][k]) for k in ['body_pose','global_orient','transl'])
            loss = (kwargs.get('penetration_weight',10)*tot_pen + kwargs.get('contact_weight',1)*tot_con) + 0.01*reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.smpl_mlp.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            tot_con_losses.append(tot_con.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pen': f"{tot_pen.item():.4f}", 'con': f"{tot_con.item():.4f}"})
        
        print(f'epoch {epoch} con loss {np.mean(tot_con_losses)}')
        return np.mean(losses)
    
    def save(self, path, epoch, opt, loss):
        torch.save({
            'epoch': epoch, 
            'mlp': self.smpl_mlp.state_dict(), 
            'opt': opt.state_dict(), 
            'loss': loss,
            'depth_scale': self.depth_scale.data
        }, path)


# ============================================================================
# 3. Test Session Evaluation Function
# ============================================================================

def evaluate_test_session(test_data, trainer, wall_obj, verts_wall_cam, device, output_dir="test_results"):
    """
    Evaluate the test session and save visualizations.
    
    Args:
        test_data: Dictionary containing test session data
        trainer: Trained ClimbingMocapTrainer
        wall_obj: Wall object
        verts_wall_cam: Wall vertices in camera coordinates
        device: Device
        output_dir: Directory to save test results
    """
    print("\n" + "="*60)
    print("EVALUATING TEST SESSION")
    print("="*60)
    
    session_name = test_data['session_name']
    print(f"Test session: {session_name}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get refined vertices using trained model
    trainer.smpl_mlp.eval()
    with torch.no_grad():
        refined_verts, refined_params = trainer.forward_pass(
            test_data['smpl_params'], 
            test_data['betas']
        )
        
        # Apply depth scale
        scale_vec = torch.stack([
            torch.ones((), device=device), 
            torch.ones((), device=device), 
            trainer.depth_scale 
        ])
        refined_verts = refined_verts * scale_vec.view(1, 1, 3)
    
    print(f"✓ Generated refined vertices: {refined_verts.shape}")
    
    # 1. Generate overlay video
    print("\n1. Generating overlay video...")
    video_path = f"{output_dir}/{session_name}_overlay.mp4"
    verify_alignment_video(
        session_data=test_data,
        wall_obj=wall_obj,
        verts_wall_cam=verts_wall_cam,
        device=device,
        output_path=video_path,
        refined_verts=refined_verts
    )
    
    # 2. Save meshes
    print("\n2. Saving meshes...")
    body_faces = trainer.body_model.faces_tensor
    save_meshes_for_sequence(
        refined_verts=refined_verts,
        wall_verts=verts_wall_cam,
        wall_faces=wall_obj.faces,
        body_faces=body_faces,
        output_dir=output_dir,
        session_name=session_name,
        device=device
    )
    
    print("\n" + "="*60)
    print(f"TEST EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}/{session_name}/")
    print("="*60)


# ============================================================================
# 4. Main Execution
# ============================================================================

def main():
    device = 'cuda'
    print("="*60 + "\nINITIALIZING WALL & ALIGNMENT\n" + "="*60)
    
    wall = Wall('single_view.ply', 'wall1.png', 'wall_mesh_segments.npy', device=device)
    verts_local = wall.forward(torch.zeros(1, wall.num_segments*2, device=device))[0]
    
    R_wc = torch.tensor([[-1, 0, 0, 0.], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], device=device).float()
    verts_h = torch.cat([verts_local, torch.ones((len(verts_local), 1), device=device)], 1)
    verts_wall_cam = (R_wc @ verts_h.T).T[:, :3]
    print(f"✓ Wall aligned. Vertices: {verts_wall_cam.shape}")

    print("\nLOADING DATA...")
    training_sessions = sorted(os.listdir('ascendmotion_merged'))
    testing_sessions = ['20240927JimeiYanwu_YYY_001']
    training_sessions = [s for s in training_sessions if s not in testing_sessions]

    st()
    
    # Load training data
    train_data = []
    for s in training_sessions:
        session_name = s.replace('_images','')
        p = f'ascendmotion_merged/{s.replace("_images","")}/merged_smpl_params.pt'
        if not os.path.exists(p): continue
        params = torch.load(p, map_location='cpu')
        params = {k: v.to(device) for k,v in params.items()}
        
        if params['body_pose'].dim() == 4: params['body_pose'] = matrix_to_axis_angle(params['body_pose'].reshape(-1,3,3)).reshape(len(params['body_pose']),-1)
        elif params['body_pose'].dim() == 3: params['body_pose'] = params['body_pose'].reshape(len(params['body_pose']),-1)
        if params['global_orient'].dim() >= 3: 
             params['global_orient'] = matrix_to_axis_angle(params['global_orient'].reshape(-1,3,3)).reshape(len(params['global_orient']),-1)

        num_frames = len(params['body_pose'])
        
        # Load LiDAR data (frame-aligned: LiDAR frame i matches SMPL frame i)
        lidar_points_by_frame = load_lidar_for_session(
            session_name, num_frames, device=device, 
            dataset_train_dir=DATASET_TRAIN_DIR, lidar_scale=LIDAR_SCALE
        )

        train_data.append({
            'smpl_params': params, 
            'betas': params.get('betas', torch.zeros(num_frames, 10, device=device)), 
            'session_name': session_name, 
            'gvhmr_K': params['K_fullimg'],
            'lidar_points_by_frame': lidar_points_by_frame
        })
        print(f"  ✓ {s} ({num_frames} frames)")

    # Load test data
    test_data = None
    for s in testing_sessions:
        session_name = s.replace('_images','')
        p = f'ascendmotion_merged/{s.replace("_images","")}/merged_smpl_params.pt'
        if not os.path.exists(p): continue
        params = torch.load(p, map_location='cpu')
        params = {k: v.to(device) for k,v in params.items()}
        
        if params['body_pose'].dim() == 4: params['body_pose'] = matrix_to_axis_angle(params['body_pose'].reshape(-1,3,3)).reshape(len(params['body_pose']),-1)
        elif params['body_pose'].dim() == 3: params['body_pose'] = params['body_pose'].reshape(len(params['body_pose']),-1)
        if params['global_orient'].dim() >= 3: 
             params['global_orient'] = matrix_to_axis_angle(params['global_orient'].reshape(-1,3,3)).reshape(len(params['global_orient']),-1)

        num_frames = len(params['body_pose'])
        
        test_data = {
            'smpl_params': params, 
            'betas': params.get('betas', torch.zeros(num_frames, 10, device=device)), 
            'session_name': session_name, 
            'gvhmr_K': params['K_fullimg'],
            'lidar_points_by_frame': None  # No LiDAR for test
        }
        print(f"  ✓ TEST: {s} ({num_frames} frames)")

    print("\n" + "="*60 + "\nSTARTING TRAINING\n" + "="*60)
    body = smplx.create(model_path="smpl_models", model_type="smpl", gender='male', batch_size=16).to(device).eval()
    loss_fn = ClimbingLoss(device=device)
    trainer = ClimbingMocapTrainer(verts_wall_cam, wall.faces, body, loss_fn, device)
    loader = DataLoader(ClimbingMocapDataset(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Init Optimizer and check for checkpoint
    first_batch = next(iter(loader))
    trainer.forward_pass(first_batch['smpl_params'], first_batch['betas'])
    
    # Initialize depth scale
    init_scale = compute_initial_depth_scale(
        data=train_data,
        body_model=body,
        verts_wall_cam=verts_wall_cam,
        device=device,
    )
    trainer.depth_scale.data[...] = init_scale
    opt = optim.Adam(
        list(trainer.smpl_mlp.parameters()) + 
        list(trainer.wall_mlp.parameters()) +
        [trainer.depth_scale],
        lr=1e-4
    )
    # Check for existing checkpoints and load the latest one
    start_epoch = 0
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('cp_*.pt'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"\n✓ Found checkpoint: {latest_checkpoint}")
            print(f"  Loading checkpoint to resume training...")
            
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            trainer.smpl_mlp.load_state_dict(checkpoint['mlp'])
            opt.load_state_dict(checkpoint['opt'])
            start_epoch = checkpoint['epoch'] + 1
            
            # Load depth_scale if it exists in checkpoint
            if 'depth_scale' in checkpoint:
                trainer.depth_scale.data[...] = checkpoint['depth_scale']
            
            print(f"  ✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"  ✓ Previous loss: {checkpoint['loss']:.6f}")
            print(f"  ✓ Continuing from epoch {start_epoch}")
        else:
            print("\n✓ No checkpoints found. Starting training from scratch.")
    else:
        print("\n✓ No checkpoint directory found. Starting training from scratch.")

    print("✓ Optimizer initialized.")
    
    # Logging
    log_path = "training_log.txt"
    with open(log_path, "w") as f: f.write("Epoch, Loss\n")
    
    Path('checkpoints').mkdir(exist_ok=True)
    
    # --- TRAINING LOOP ---
    for ep in range(start_epoch, 50):
        loss = trainer.train_epoch(loader, opt, ep, contact_weight=1.0, penetration_weight=10.0)
        print(f"  Ep {ep}: {loss:.4f}")
        
        with open(log_path, "a") as f: f.write(f"{ep}, {loss:.6f}\n")
        
        if (ep+1)%10==0: 
            trainer.save(f'checkpoints/cp_{ep+1}.pt', ep, opt, loss)

        if args.viz_overlay is True:        
            # --- VIDEO SAVING EVERY 20 ITERATIONS ---
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"\nCreating visualization for Epoch {ep}...")
                # Pick the first session (train_data[0]) to visualize consistency
                vis_sample = train_data[0]
                with torch.no_grad():
                    # Get REFINED vertices using the current trained MLP
                    refined_verts, _ = trainer.forward_pass(vis_sample['smpl_params'], vis_sample['betas'])
                    
                verify_alignment_video(
                    session_data=vis_sample, 
                    wall_obj=wall, 
                    verts_wall_cam=verts_wall_cam, 
                    device=device,
                    output_path=f"checkpoints/vis_epoch_{ep}.mp4",
                    refined_verts=refined_verts
                )
            # ----------------------------------------

    # --- TEST SESSION EVALUATION ---
    if test_data is not None:
        print("\n" + "="*60)
        print("RUNNING TEST SESSION EVALUATION")
        print("="*60)
        evaluate_test_session(
            test_data=test_data,
            trainer=trainer,
            wall_obj=wall,
            verts_wall_cam=verts_wall_cam,
            device=device,
            output_dir="test_results"
        )
    else:
        print("\n⚠ No test data found - skipping test evaluation")

if __name__ == "__main__":
    main()