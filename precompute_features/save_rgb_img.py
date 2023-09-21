import os
import sys
import math
import json
from tqdm import tqdm
import numpy as np
import h5py
from progressbar import ProgressBar
import torch.multiprocessing as mp
import argparse

sys.path.insert(0, 'build')
import MatterSim

WIDTH = 224
HEIGHT = 224
VFOV = 90

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]      # load all scans
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def get_img(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)
    
    for scan_id, viewpoint_id in scanvp_list:
        images = []
        for ix in range(12):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            image = np.array(state.rgb, copy=True)      # in BGR channel
            images.append(image)
        images = np.stack(images, axis=0)
        out_queue.put((scan_id, viewpoint_id, images))

    out_queue.put(None)

def build_img_file(args):

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
            target=get_img,
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
                scan_id, viewpoint_id, images = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                outf.create_dataset(key, data=images, dtype='uint8', compression='gzip')

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='precompute_features/connectivity')
    parser.add_argument('--scan_dir', default='/data/andong/Matterport/v1/scans') # mp3d scan path
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    args.output_file = f'img_features/mp3d_{HEIGHT}x{WIDTH}_vfov{VFOV}_bgr.hdf5'

    build_img_file(args)