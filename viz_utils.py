import pytorch3d
import torch
import numpy as np
import os
from pdb import set_trace as st
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    SoftPhongShader
)

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import smplx
import cv2
import imageio
from tqdm import tqdm

def SMPL_to_mesh_single(smpl_params, body_model, device):
    """Create mesh for a single frame"""
    params = dict(
        global_orient=smpl_params['global_orient'],
        body_pose=smpl_params['body_pose'],        
        betas=smpl_params['betas'],
        transl=smpl_params['trans'],
        return_verts=True,
    )
    
    with torch.no_grad():
        out = body_model(**params)
    
    verts = out.vertices.to(device)
    faces = torch.from_numpy(body_model.faces.astype(np.int64)).to(device)
    faces = faces.unsqueeze(0).repeat(verts.shape[0], 1, 1)
    texture = torch.ones_like(verts).to(device)
    
    mesh = pytorch3d.structures.Meshes(
        verts, faces, 
        textures=pytorch3d.renderer.TexturesVertex(texture)
    ).to(device)
    
    return mesh



def viz_scene(smpl_seq=None, point_clouds = None, images = None, T_w2c=None, 
              savepath='scene.mp4', save_options = ['SMPL', 'points'], overlay_images = True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if images is not None:
        Hbg, Wbg = imageio.imread(images[0]).shape[:2]
    else:
        Hbg, Wbg = 512, 512

    FPS = 20
    
    # stream to file
    if 'SMPL' in save_options:
        writer = imageio.get_writer(savepath.replace('.mp4', '_SMPL.mp4'), fps=FPS, codec="libx264", quality=8)


    if smpl_seq is not None:
        N_frames = smpl_seq['body_pose'].shape[0]
        
        # Create body model
        body_model = smplx.create(
            model_path="smpl_models/models",
            model_type="smpl",
            gender='male',      
            use_pca=True,      
            ext="pkl",
            batch_size=1
        ).to(device)
        
        # Build camera and lights
        camera, light = build_stationary_camera(T_w2c, image_size =(Hbg, Wbg))
        camera = camera.to(device)
        light = light.to(device)
        
        # Process each frame individually
        tqdm_range = tqdm(range(N_frames), desc="Rendering SMPL Sequence")
        for i in tqdm_range:
            # Extract parameters for single frame
            frame_params = {
                'global_orient': smpl_seq['global_orient'][i:i+1],
                'body_pose': smpl_seq['body_pose'][i:i+1],
                'betas': smpl_seq['betas'][i:i+1] if smpl_seq['betas'].dim() > 1 else smpl_seq['betas'].unsqueeze(0),
                'trans': smpl_seq['trans'][i:i+1],
            }
            
            with torch.no_grad():
                # Create mesh for single frame
                mesh = SMPL_to_mesh_single(frame_params, body_model, device)
                

                # Create renderer for this frame
                renderer = get_mesh_renderer(image_size=(Hbg, Wbg), lights=light, device=device)


                # Render single frame
                img = renderer(mesh, cameras=camera, lights=light)
                frame = img[0].detach().cpu().numpy()


                # Overlay with image if provided
                if overlay_images and images is not None:
                    img_path = images[i]
                    img = imageio.imread(img_path)
                    img = (img / 255.).astype(np.float32)
                    # Add in alpha channel
                    if img.ndim == 3:
                        alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                        img = np.concatenate([img, alpha_channel], axis=-1)
                    frame = 0.5 * frame + 0.5 * img

                # Write frame to video
                if 'SMPL' in save_options and writer is not None:
                    writer.append_data((frame[..., :3] * 255).clip(0, 255).astype(np.uint8))

                # Clean up
                del mesh, img, renderer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Clean up body model
        del body_model
        if device.type == "cuda":
            torch.cuda.empty_cache()



    if point_clouds is not None:
        point_clouds = point_clouds.to(device)

        N_frames = point_clouds.shape[0]
        cameras_all, _ = build_stationary_camera(N_frames, T_w2c)
        cameras_all = cameras_all.to(device)

        points_renderer = get_points_renderer(image_size=512, device=device, radius=0.01, background_color=(1,1,1))

        # constant blue per point
        blue = torch.tensor([0.43, 0.71, 1.0], device=device, dtype=torch.float32)

        frames = []
        with torch.no_grad():
            for i in tqdm(range(N_frames), desc="Rendering Point Clouds"):
                pts_i = point_clouds[i]                                      # (P,3)
                feat_i = blue.expand(pts_i.shape[0], 3)                      # (P,3)
                pcd_i = pytorch3d.structures.Pointclouds(points=[pts_i], features=[feat_i]).to(device)

                cam_i = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=cameras_all.R[i:i+1], T=cameras_all.T[i:i+1], fov=60, device=device
                )

                img = points_renderer(pcd_i, cameras=cam_i)                  # (1,H,W,4 or 3)
                frames.append(img[0, ..., :3].detach().cpu().numpy())        # (H,W,3)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        frames_pts = np.stack(frames, axis=0)
        make_gif(frames_pts, N_frames, savepath=savepath.replace('.mp4', '_points.mp4'))



    # if 'SMPL' in save_options:
    #     if overlay_images:
    #         assert images is not None, "images must be provided for overlay"
    #         assert frames_SMPL.shape[0] == frames_img.shape[0], "Number of frames in SMPL and images must match"
    #         alpha = 0.5
    #         frames_SMPL = alpha * frames_SMPL + (1 - alpha) * frames_img

    #     make_gif(frames_SMPL, N_frames, savepath=savepath.replace('.mp4', '_SMPL.mp4'))

    # if 'points' in save_options:
    #     make_gif(frames_pts, N_frames, savepath=savepath.replace('.mp4', '_points.mp4'))


@torch.inference_mode()
def viz_scene_batched(smpl_seq=None, images = None, T_w2c=None, 
              savepath='scene.mp4', batch_size=8, FPS = 20, downsizing_factor=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if images is not None:
        Hbg, Wbg = imageio.imread(images[0]).shape[:2]

    else:
        Hbg, Wbg = 512, 512


    
    # stream to file
    writer = imageio.get_writer(savepath.replace('.mp4', '_SMPL.mp4'), fps=FPS, codec="libx264", quality=8)


    # Create body model and get verts and faces
    body_model = smplx.create(
        model_path="smpl_models/models",
        model_type="smpl",
        gender='male',      
        use_pca=True,      
        ext="pkl",
        batch_size=1
    ).to(device)
    with torch.no_grad():
        out = body_model(**smpl_seq)
    verts = out.vertices.to(device)
    if 'faces' not in locals():
        faces = torch.from_numpy(body_model.faces.astype(np.int64)).to(device)
        faces = faces.unsqueeze(0).repeat(verts.shape[0], 1, 1)
        texture = torch.ones_like(verts).to(device)
    else: pass
    
    # Build camera and lights
    camera, light = build_stationary_camera(T_w2c, image_size =(Hbg, Wbg))
    camera = camera.to(device)
    light = light.to(device)

    # define mesh renderer
    renderer = get_mesh_renderer(image_size=(Hbg, Wbg), lights=light, device=device, cameras = camera)
    
    # Process batched frames without rendering everything at once
    N_frames = smpl_seq['body_pose'].shape[0]
    tqdm_range = tqdm(range(0, N_frames, batch_size), desc="Rendering SMPL Sequence")
    for i in tqdm_range:    
        # Create mesh for single frame
        mesh = pytorch3d.structures.Meshes(
            verts[i:i+batch_size], faces[i:i+batch_size], 
            textures=pytorch3d.renderer.TexturesVertex(texture[i:i+batch_size])
        ).to(device)

        # Create renderer for this frame
        rend = renderer(mesh, cameras=camera, lights=light).detach()
        del mesh
        
        # Overlay with images
        img_path = images[i:i+batch_size]
        img = [imageio.imread(p) for p in img_path]
        img = [(im / 255.).astype(np.float32) for im in img]
        # Add in alpha channel
        for j in range(len(img)):
            alpha_channel = np.ones((img[j].shape[0], img[j].shape[1], 1), dtype=img[j].dtype)
            img[j] = np.concatenate([img[j], alpha_channel], axis=-1)
        img = np.stack(img, axis=0)  # (B,H,W,4)
        img = torch.from_numpy(img).to(device=device, dtype=torch.float32)

        # Alpha composite on GPU
        alpha = rend[..., 3:4]
        frames = rend[..., :3] * alpha + img[..., :3] * (1 - alpha)
        frames = frames.clamp(0, 1).cpu().numpy()  # (B,H,W,3)

        # Write frame to video
        for frame in frames:
            frame = (frame[..., :3] * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.resize(frame, (Wbg//downsizing_factor, Hbg//downsizing_factor), interpolation=cv2.INTER_AREA)
            writer.append_data(frame)
    writer.close()




def build_stationary_camera(T_w2c, image_size, camera_matrix = np.array([[2105.399,0.000,970.658],[0.000, 2105.376, 621.038],[0.000,0.000,1.000]], dtype=np.float32)):
    # --- Extract rotation and translation (world→camera)
    R = torch.from_numpy(T_w2c[:3, :3]).float().unsqueeze(0)  # (1,3,3)
    t = torch.from_numpy(T_w2c[:3, 3]).float().unsqueeze(0)   # (1,3)

    # --- Build pixel-space camera (OpenCV convention)
    cameras = cameras_from_opencv_projection(
        R=R,
        tvec=t,
        camera_matrix=torch.from_numpy(camera_matrix).float().unsqueeze(0),
        image_size=torch.tensor([image_size], dtype=torch.float32)
    )

    # --- Compute camera center in world coords for lighting
    C_world = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)
    lights = pytorch3d.renderer.PointLights(location=C_world)

    return cameras, lights


def viz_pointcloud(pointclouds, savepath = 'pointcloud.gif'):

    blue = torch.tensor([0.43, 0.71, 1], device=pointclouds.device)
    features = blue.expand(pointclouds.shape[0], pointclouds.shape[1], 3)
    point_clouds = pytorch3d.structures.Pointclouds(points=pointclouds, features=features)

    cameras, _ = build_360_camera(N_frames, distance = 2)
    renderer = get_points_renderer(
            image_size=512, background_color=(1,1,1)
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_clouds = point_clouds.to(device)
    cameras = cameras.to(device)

    rend = renderer(point_clouds, cameras=cameras)
    rend = rend.cpu().numpy()[:, ..., :3] 
    make_gif(rend, N_frames, savepath=savepath)



def build_360_camera(N_frames, distance = 3, elev = 30):
    # define camera with 360 degree rotation
    azims = torch.linspace(0, 360, N_frames)
    dists = torch.ones_like(azims) * distance
    elevs = torch.ones_like(azims) * elev

    R,T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist = dists, # distance of the camera from the object
            elev = elevs, # elevation angle in degrees
            azim = azims, # azimuth angle in degrees
        )

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R = R,
        T = T,
        fov = 60,
        )


    # render and save images
    # make point light to rotate with the camera
    T = torch.linspace(0, 360, N_frames)
    T = torch.stack([torch.sin(T * np.pi / 180) * 3, torch.ones_like(T) * 2, torch.cos(T * np.pi / 180) * 3], dim=1)
    lights = pytorch3d.renderer.PointLights(location=T) # let the light be at the camera location 

    return cameras, lights

def make_gif(rend, N_frames, savepath = 'out.gif', loop = 0):
    
    import imageio
    frames = []
    for i in range(rend.shape[0]):
        img = rend[i, ..., :3]              # (H, W, 3), float32 in [0,1]
        img = (img * 255).clip(0, 255)      # scale to [0,255]
        img = img.astype(np.uint8)          # convert to uint8
        frames.append(img)
    duration = N_frames // 2  # Convert FPS (frames per second) to duration (ms per frame)

    if savepath.endswith('.gif'):
        imageio.mimsave(savepath, frames, duration=duration/1000, loop=loop)
    else:
        imageio.mimsave(savepath, frames, fps=30)

import torch
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardPhongShader, SoftPhongShader,
    RasterizationSettings, BlendParams
)

def get_mesh_renderer(
    cameras,
    image_size=(512, 512),          # (H, W)
    lights=None,
    device=None,
    soft=False,                      # True -> SoftPhong (smoother), False -> HardPhong (faster)
    faces_per_pixel=1,               # >1 gives AA/soft edges but costs time
    cull_backfaces=True,
    ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rast_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=cull_backfaces,
        blur_radius=0.0,             # keep 0.0 for speed (set >0 only if using soft edges)
        bin_size=0,                  # auto-tune tiling on GPU
        max_faces_per_bin=0          # auto-tune
    )

    blend = BlendParams(background_color=(0.0, 0.0, 0.0))

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rast_settings)
    Shader = SoftPhongShader if soft else HardPhongShader
    shader = Shader(device=device, cameras=cameras, lights=lights, blend_params=blend)

    return MeshRenderer(rasterizer=rasterizer, shader=shader)

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer



# %% This is copied from source code of pytorch3d
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.camera_conversions import (
    _cameras_from_opencv_projection,
    _opencv_from_cameras_projection,
    _pulsar_from_cameras_projection,
    _pulsar_from_opencv_projection,
)

def cameras_from_opencv_projection(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
) -> PerspectiveCameras:
    """
    Converts a batch of OpenCV-conventioned cameras parametrized with the
    rotation matrices `R`, translation vectors `tvec`, and the camera
    calibration matrices `camera_matrix` to `PerspectiveCameras` in PyTorch3D
    convention.

    More specifically, the conversion is carried out such that a projection
    of a 3D shape to the OpenCV-conventioned screen of size `image_size` results
    in the same image as a projection with the corresponding PyTorch3D camera
    to the NDC screen convention of PyTorch3D.

    More specifically, the OpenCV convention projects points to the OpenCV screen
    space as follows::

        x_screen_opencv = camera_matrix @ (R @ x_world + tvec)

    followed by the homogenization of `x_screen_opencv`.

    Note:
        The parameters `R, tvec, camera_matrix` correspond to the inputs of
        `cv2.projectPoints(x_world, rvec, tvec, camera_matrix, [])`,
        where `rvec` is an axis-angle vector that can be obtained from
        the rotation matrix `R` expected here by calling the `so3_log_map` function.
        Correspondingly, `R` can be obtained from `rvec` by calling `so3_exp_map`.

    Args:
        R: A batch of rotation matrices of shape `(N, 3, 3)`.
        tvec: A batch of translation vectors of shape `(N, 3)`.
        camera_matrix: A batch of camera calibration matrices of shape `(N, 3, 3)`.
        image_size: A tensor of shape `(N, 2)` containing the sizes of the images
            (height, width) attached to each camera.

    Returns:
        cameras_pytorch3d: A batch of `N` cameras in the PyTorch3D convention.
    """
    return _cameras_from_opencv_projection(R, tvec, camera_matrix, image_size)

# %%
