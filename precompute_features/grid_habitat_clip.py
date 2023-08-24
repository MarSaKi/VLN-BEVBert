import os
import sys
import argparse
import numpy as np
import json
import math
import h5py
from PIL import Image
import cv2
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
sys.path.append('.')
import precompute_features.clip as clip

sys.path.append('precompute_features/build_cpu')
import MatterSim
import habitat
from precompute_features.utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R


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

# TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 12 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 224
HEIGHT = 224
VFOV = 90
HFOV = 90

def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device

def build_simulator(connectivity_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def build_habitat_sim(scan):
    sim = HabitatUtils(f'/data/andong/vlnce_dataset/data/scene_datasets/mp3d/{scan}/{scan}.glb', int(0), int(math.degrees(HFOV)), HEIGHT, WIDTH)
    return sim

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir)
    habitat_sim = None
    pre_scan_id = None

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name)

    for scan_id, viewpoint_id in scanvp_list:
        if scan_id != pre_scan_id:
            if habitat_sim != None:
                habitat_sim.sim.close()
            habitat_sim = build_habitat_sim(scan_id)
        pre_scan_id = scan_id

        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            # set habitat to the same position & rotation
            x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
            habitat_position = [x, z-1.25, -y]
            mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
            mp3d_e = np.array([e, 0, 0])
            rotvec_h = R.from_rotvec(mp3d_h)
            rotvec_e = R.from_rotvec(mp3d_e)
            habitat_rotation = (rotvec_h * rotvec_e).as_quat()
            habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

            image = np.array(habitat_sim.render('rgb'), copy=True)  # in RGB channel
            image = Image.fromarray(image)  # input RGB
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0).cuda() # 12 x 3 x 224 x 224
        grid_fts = []
        for k in range(0, len(images), args.batch_size):
            _, b_grid_fts = model.encode_image(images[k: k+args.batch_size])
            b_grid_fts = b_grid_fts.data.cpu().numpy()
            grid_fts.append(b_grid_fts)
        grid_fts = np.concatenate(grid_fts, 0).astype(np.float16)

        out_queue.put((scan_id, viewpoint_id, grid_fts))

    out_queue.put(None)


def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True) 

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, grid_fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                data = grid_fts # 12 x 196 x 768
                outf.create_dataset(key, data=data, dtype='float16', compression='gzip')

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == '__main__':
    os.system("mkdir -p img_features")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ViT-B/16')
    parser.add_argument('--connectivity_dir', default='precompute_features/connectivity')
    parser.add_argument('--output_file', default='img_features/vit_b16_224_clip_patch_habitat.hdf5')
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', type=int, default=1) 
    args = parser.parse_args()

    build_feature_file(args)