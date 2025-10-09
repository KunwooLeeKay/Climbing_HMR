import os
from pdb import set_trace as st
import pickle

import torch
import pytorch3d
import numpy as np

from viz_utils import *

import argparse

parser = argparse.ArgumentParser(description='Visualize climbing mocap data')
parser.add_argument('--seq_idx', type=int, help='Sequence index', default=0)
parser.add_argument('--person', type=str, help='Dataset index', default='first_person')
parser.add_argument('--downsample', action='store_true', help='Downsample the sequence by a factor of 2')
parser.add_argument('--trim', action='store_true', help='Trim the sequence to the middle third')
parser.add_argument('--use_opt_pose', action='store_true', help='Use optimized pose inside dataset. Only available when second_person')
parser.add_argument('--use_opt_trans', action='store_true', help='Use optimized translation inside dataset. Only available when second_person')
args = parser.parse_args()



def main():
    dataset_dir = '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1'

    os.listdir(dataset_dir)

    TRAIN = os.path.join(dataset_dir, 'Dataset_Train')
    TRAIN_2D = os.path.join(dataset_dir, 'Dataset_Train_2D')

    dataset_name_list = os.listdir(TRAIN)
    npz_names = [f for f in dataset_name_list if f.endswith('.npz')]
    pkl_names = [f for f in dataset_name_list if f.endswith('.pkl')]


    SEQ_IDX = args.seq_idx
    print(f'Visualizing sequence {SEQ_IDX}: {pkl_names[SEQ_IDX]}')
    data = pickle.load(open(os.path.join(TRAIN, pkl_names[SEQ_IDX]), 'rb'))

    # NOTE All of this is for person. Not for the wall.
    if args.use_opt_pose:
        assert args.person == 'second_person', "use_opt is only available when second_person"
        print("Using optimized pose")
        pose = data[args.person]['opt_pose'][:, 3:] # first 3 are global orient
        global_orient = data[args.person]['opt_pose'][:, :3]
        trans = data[args.person]['opt_trans'] if args.use_opt_trans else data[args.person]['mocap_trans']
        gender = data[args.person]['gender']
    else:
        pose = data[args.person]['pose'][:, 3:] # first 3 are global orient
        global_orient = data[args.person]['pose'][:, :3]
        trans = data[args.person]['mocap_trans']

    betas = np.tile(np.array(data[args.person]['beta']), (pose.shape[0], 1))  # (N, 10)

    # The reconstructed wall -> .ply file
    wall = None

    # Overlay the RGB video
    video_path = os.path.join(TRAIN_2D, npz_names[SEQ_IDX].replace('_W2C.npz', '_images'))
    images = os.listdir(video_path)
    images = [f for f in images if f.endswith('.jpg')]
    images = [os.path.join(video_path, f) for f in images]


    if args.downsample:
        # downsample
        downsample_factor = 2
        pose = pose[::downsample_factor]
        global_orient = global_orient[::downsample_factor]
        trans = trans[::downsample_factor]
        betas = betas[::downsample_factor]
        images = images[::downsample_factor]
    elif args.trim:
        # trim
        trim_start = len(pose) // 6 * 2
        trim_end = len(pose) // 6 * 3
        print(f'Trimming to frames {trim_start} to {trim_end}, which is {trim_end - trim_start} frames in total. orginal length was {len(pose)}')
        print(f'Trimming to {trim_end - trim_start} / {len(pose)} frames')
        pose = pose[trim_start:trim_end]
        global_orient = global_orient[trim_start:trim_end]
        trans = trans[trim_start:trim_end]
        betas = betas[trim_start:trim_end]
        images = images[trim_start:trim_end]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smpl_seq = {
        'body_pose': torch.from_numpy(pose).float().to(device),
        'global_orient': torch.from_numpy(global_orient).float().to(device),
        'betas': torch.from_numpy(betas).float().to(device),
        'trans': torch.from_numpy(trans).float().to(device),
    }

    # import W2C transform
    T_w2c = np.load(os.path.join(TRAIN, npz_names[SEQ_IDX]))['T_w2c']

    savedir = f"output_viz/{args.person}/{'opt_pose' if args.use_opt_pose else 'mocap_pose'}/{'opt_trans' if args.use_opt_trans else 'mocap_trans'}"
    os.makedirs(savedir, exist_ok=True)
    viz_scene_batched(smpl_seq = smpl_seq, images = images, T_w2c = T_w2c, savepath = f"{savedir}/{pkl_names[SEQ_IDX].replace('.pkl', '.mp4')}")


if __name__ == '__main__':
    main()