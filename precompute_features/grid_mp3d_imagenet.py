#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys
sys.path.insert(0, 'build')
import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
import msgpack_numpy
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 12 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 224
HEIGHT = 224
VFOV = 90

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

def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    config['crop_pct'] = 1.0
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setRenderingEnabled(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def get_patch_fts(model, img_transforms, device, images):

    def hook_builder(tgt_tensor):
        def hook(m, i, o):
            tgt_tensor.set_(o)
        return hook

    patch_fts = torch.zeros((1,), device=device)
    patch_hook = model.norm.register_forward_hook(hook_builder(patch_fts))
    with torch.no_grad():
        global_fts = model.forward_features(images)
    patch_fts = patch_fts[:, 1:, :]

    return patch_fts

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            # elif ix % 12 == 0:
            #     sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0) # 12 x 3 x 224 x 224
        # fts, logits = [], []
        fts = []
        for k in range(0, len(images), args.batch_size):
            # b_fts = model.forward_features(images[k: k+args.batch_size])
            # b_logits = model.head(b_fts)
            b_fts = get_patch_fts(model, img_transforms, device, images[k: k+args.batch_size]) # 12 x 196 x 768
            b_fts = b_fts.data.cpu().numpy()
            # b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            # logits.append(b_logits)
        fts = np.concatenate(fts, 0).astype(np.float16)
        # logits = np.concatenate(logits, 0)
        logits = None

        out_queue.put((scan_id, viewpoint_id, fts, logits))

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
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                if args.out_image_logits:
                    data = np.hstack([fts, logits])
                else:
                    data = fts # 12 x 196 x 768
                outf.create_dataset(key, data=data, dtype='float16', compression='gzip')

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='precompute_features/connectivity')
    parser.add_argument('--scan_dir', default='/data/andong/Matterport/v1/scans') # mp3d scan path
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default='img_features/vit_b16_224_imagenet_patch.hdf5')
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)