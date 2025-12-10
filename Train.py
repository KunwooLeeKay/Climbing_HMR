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

# Make argparse object
parser = argparse.ArgumentParser(description='Run Training')
parser.add_argument('--viz_overlay', type=int, help='Sequence index', default=False)
args = parser.parse_args()



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
    img_dir = f'/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train_2D/{session_name}_images'
    images = sorted(os.listdir(img_dir))
    
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
        bg = cv2.imread(os.path.join(img_dir, images[i]))
        if bg is None: continue
        
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
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            opt.zero_grad()
            if not self.smpl_mlp: 
                self.forward_pass(batch['smpl_params'], batch['betas'])
                opt.add_param_group({'params': self.smpl_mlp.parameters(), 'lr': 1e-4})
                self.smpl_mlp.train()
            
            verts, params_ref = self.forward_pass(batch['smpl_params'], batch['betas'])
            
            tot_pen = 0; tot_con = 0; tot_loss = 0
            chunk = 64
            for i in range(0, len(verts), chunk):
                v_chunk = verts[i:i+chunk]
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
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'pen': f"{tot_pen.item():.4f}"})
            
        return np.mean(losses)
    
    def save(self, path, epoch, opt, loss):
        torch.save({'epoch': epoch, 'mlp': self.smpl_mlp.state_dict(), 'opt': opt.state_dict(), 'loss': loss}, path)


# ============================================================================
# 3. Main Execution
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
    training_sessions = training_sessions[:-1]
    
    data = []
    for s in training_sessions:
        p = f'ascendmotion_merged/{s.replace("_images","")}/merged_smpl_params.pt'
        if not os.path.exists(p): continue
        params = torch.load(p, map_location='cpu')
        params = {k: v.to(device) for k,v in params.items()}
        
        if params['body_pose'].dim() == 4: params['body_pose'] = matrix_to_axis_angle(params['body_pose'].reshape(-1,3,3)).reshape(len(params['body_pose']),-1)
        elif params['body_pose'].dim() == 3: params['body_pose'] = params['body_pose'].reshape(len(params['body_pose']),-1)
        if params['global_orient'].dim() >= 3: 
             params['global_orient'] = matrix_to_axis_angle(params['global_orient'].reshape(-1,3,3)).reshape(len(params['global_orient']),-1)

        data.append({'smpl_params': params, 'betas': params.get('betas', torch.zeros(len(params['body_pose']), 10, device=device)), 'session_name': s.replace("_images",""), 'gvhmr_K': params['K_fullimg']})
        print(f"  ✓ {s}")

    print("\n" + "="*60 + "\nSTARTING TRAINING\n" + "="*60)
    body = smplx.create(model_path="smpl_models", model_type="smpl", gender='male', batch_size=16).to(device).eval()
    loss_fn = ClimbingLoss(device=device)
    trainer = ClimbingMocapTrainer(verts_wall_cam, wall.faces, body, loss_fn, device)
    loader = DataLoader(ClimbingMocapDataset(data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Init Optimizer
    first_batch = next(iter(loader))
    trainer.forward_pass(first_batch['smpl_params'], first_batch['betas'])
    opt = optim.Adam(trainer.smpl_mlp.parameters(), lr=1e-4)
    print("✓ Optimizer initialized.")
    
    # Logging
    log_path = "training_log.txt"
    with open(log_path, "w") as f: f.write("Epoch, Loss\n")
    
    Path('checkpoints').mkdir(exist_ok=True)
    
    # --- TRAINING LOOP ---
    for ep in range(50):
        loss = trainer.train_epoch(loader, opt, ep, contact_weight=1.0, penetration_weight=10.0)
        print(f"  Ep {ep}: {loss:.4f}")
        
        with open(log_path, "a") as f: f.write(f"{ep}, {loss:.6f}\n")
        
        if (ep+1)%10==0: trainer.save(f'checkpoints/cp_{ep+1}.pt', ep, opt, loss)

        if args.viz_overlay is True:        
            # --- VIDEO SAVING EVERY 20 ITERATIONS ---
            if (ep + 1) % 20 == 0:
                print(f"\nCreating visualization for Epoch {ep}...")
                # Pick the first session (data[0]) to visualize consistency
                vis_sample = data[0]
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

if __name__ == "__main__":
    main()