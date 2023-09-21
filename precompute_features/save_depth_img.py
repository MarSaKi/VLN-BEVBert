import os
import sys

import habitat

sys.path.append('build_cpu')    # compile mp3d sim in CPU mode
sys.path.append('.')
import MatterSim
import time
import math
import cv2
import numpy as np
import json
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn

from utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

WIDTH = 224 #480
HEIGHT = 224 #480
VFOV = math.radians(90)
HFOV = VFOV*WIDTH/HEIGHT
MAX_DEPTH = 10

''' Prepare viewpoints '''
def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

''' MP3D Sim '''
def build_mp3d_sim():
    sim = MatterSim.Simulator()
    sim.setNavGraphPath('precompute_features/connectivity')
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setRenderingEnabled(False)
    sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.setDiscretizedViewingAngles(True)
    sim.setRestrictedNavigation(False)
    sim.initialize()
    return sim

''' Habitat Sim '''
def build_habitat_sim(scan):
    sim = HabitatUtils(f'/data/andong/vlnce_dataset/data/scene_datasets/mp3d/{scan}/{scan}.glb', int(0), int(math.degrees(HFOV)), HEIGHT, WIDTH)
    return sim

def masked_nonzero_pooling(img, mask_input=None):
        """
        Avgpool over only non-zero values of the tensor

        Args:
            img (torch.FloatTensor)

        Returns:
            pooled_img (torch.FloatTensor)

        Shape:
            Input:
                img: (batch_size, feature_size, height, width)
            Output:
                pooled_img: (batch_size, feature_size, feature_map_height, feature_map_width)

        Logic:
            MaskedAvgPool = AvgPool(img) / AvgPool(img != 0)
            Denominator for both average pools cancels out.
        """
        pooler = nn.AdaptiveAvgPool2d((HEIGHT//16, WIDTH//16))

        img = torch.from_numpy(img)
        if type(mask_input) == type(None):
            mask_input = img
        pooled_img = pooler(img)
        pooled_mask = pooler((mask_input!=0).float())
        nonzero_mask = torch.where(pooled_mask == 0, torch.ones_like(pooled_mask)*1e-8, pooled_mask)
        pooled_img = torch.div(pooled_img, nonzero_mask)
        pooled_img = pooled_img.numpy()
        return pooled_img

def get_depth(mp3d_sim, habitat_sim, scan, viewpoint):
    mp3d_sim.newEpisode([scan], [viewpoint], [0], [0])
    result = np.zeros([12, HEIGHT, WIDTH], dtype=np.float32)
    for idx in range(12):
        state = mp3d_sim.getState()[0]

        # set habitat to the same position & rotation
        x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
        habitat_position = [x, z-1.25, -y]
        mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
        mp3d_e = np.array([e, 0, 0])
        rotvec_h = R.from_rotvec(mp3d_h)
        rotvec_e = R.from_rotvec(mp3d_e)
        habitat_rotation = (rotvec_h * rotvec_e).as_quat()
        habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

        # get obs
        depth = habitat_sim.render('depth')
        result[idx] = depth[:, :, 0]    # 224 x 224

        mp3d_sim.makeAction([0], [1.0], [0])

    return result

if __name__ == '__main__':
    os.system("mkdir -p img_features")
    scanvp_list = load_viewpoint_ids('precompute_features/connectivity')
    pre_scan = None
    mp3d_sim = build_mp3d_sim()
    habitat_sim = None

    with h5py.File(f'img_features/habitat_224x224_vfov90_depth.hdf5', 'w') as outf:
        for (scan, viewpoint) in tqdm(scanvp_list, total=len(scanvp_list)):
            if scan != pre_scan:
                if habitat_sim != None:
                    habitat_sim.sim.close()
                habitat_sim = build_habitat_sim(scan)
            pre_scan = scan

            depth_item = get_depth(mp3d_sim, habitat_sim, scan, viewpoint)
            outf.create_dataset(f'{scan}_{viewpoint}', data=depth_item, dtype='float32', compression='gzip')
