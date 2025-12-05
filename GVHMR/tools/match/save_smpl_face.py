# save once as hmr4d/utils/body_model/smpl_faces.npy
import numpy as np
from hmr4d.utils.smplx_utils import make_smplx
faces_smpl = make_smplx("smpl").faces  # (Fs,3) int
np.save("hmr4d/utils/body_model/smpl_faces.npy", faces_smpl)
