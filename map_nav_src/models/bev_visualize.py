import os
import sys
sys.path.insert(0, 'build')
import argparse
import h5py 
import cv2
import json
import networkx as nx
import numpy as np
import MatterSim
import torch
from tqdm import tqdm
from torch_scatter import scatter_max
from .bev_utils import transfrom3D, PointCloud

MAP_DIM = 301 
MAP_RES = 1/30
WIDTH = 224
HEIGHT = 224
VFOV = np.radians(90)
HFOV = VFOV*WIDTH/HEIGHT

sim = MatterSim.Simulator()
sim.setNavGraphPath('datasets/R2R/connectivity')
sim.setCameraResolution(224, 224)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False)
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setRestrictedNavigation(False)
sim.initialize()
projector = PointCloud(VFOV, 1, 224, 224, 321, 321, torch.FloatTensor([0,0,0]), 0.5, torch.device('cpu'))
rgb_db = h5py.File('img_features/mp3d_224x224_vfov90_bgr.hdf5', 'r')
depth_db = h5py.File('img_features/habitat_224x224_vfov90_depth.hdf5', 'r')

'''
MP3D original semantic labels
# Original set from here: https://github.com/niessner/Matterport/blob/master/metadata/mpcat40.tsv
0 void 0
1 wall 15 structure
2 floor 17 free-space
3 chair 1
4 door 2
5 table 3
6 picture 18
7 cabinet 19
8 cushion 4
9 window 15 structure
10 sofa 5
11 bed 6
12 curtain 16 other
13 chest_of_drawers 20
14 plant 7
15 sink 8
16 stairs 17 free-space
17 ceiling 17 free-space
18 toilet 9
19 stool 21
20 towel 22
21 mirror 16 other
22 tv_monitor 10
23 shower 11
24 column 15 structure
25 bathtub 12
26 counter 13
27 fireplace 23
28 lighting 16 other
29 beam 16 other
30 railing 16 other
31 shelving 16 other
32 blinds 16 other
33 gym_equipment 24
34 seating 25
35 board_panel 16 other
36 furniture 16 other
37 appliances 14
38 clothes 26
39 objects 16 other
40 misc 16 other
'''

def build_sim():
    sim = MatterSim.Simulator()
    sim.setNavGraphPath('datasets/R2R/connectivity')
    sim.setCameraResolution(224, 224)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(False)
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setRestrictedNavigation(False)
    sim.initialize()
    return sim 

def lift(projector, state, rgb_db, depth_db, return_img=False):
    scan = state.scanId
    vp = state.location.viewpointId
    scan_vp = f'{scan}_{vp}'
    viewidx = state.viewIndex

    xyzhe = np.zeros([12,5])
    xyzhe[:,0] = state.location.x
    xyzhe[:,1] = state.location.z
    xyzhe[:,2] = -state.location.y
    xyzhe[:,3] = -(np.arange(12) * np.radians(30) + state.heading) # counter-clock
    xyzhe[:,4] = np.pi
    T = torch.from_numpy(transfrom3D(xyzhe))

    roll_idx = np.arange(12)
    roll_idx = np.roll(roll_idx, -(viewidx%12))
    rgbs = rgb_db[scan_vp][...].astype(np.uint8)[roll_idx]
    depths = depth_db[scan_vp][...].astype(np.float32)[roll_idx]
    if return_img:
        img = np.concatenate([rgbs[-3], rgbs[0], rgbs[3]], axis=1)
    
    depths_var = torch.from_numpy(depths*10)[:,None,:,:]
    pc, pc_mask = projector.forward(depths_var, T)
    pc = pc.reshape(-1, 3).numpy()
    pc_mask = pc_mask.reshape(-1).numpy()
    pc_feat = rgbs.reshape(-1, 3)

    out = (pc, pc_feat, pc_mask, )
    if return_img:
        out = (pc, pc_feat, pc_mask, img, )
    return out

def splat(state, pc_hist, pc_feat_hist, pc_mask_hist, fuse_step=1):
    pc = pc_hist[-fuse_step:]
    pc_feat = pc_feat_hist[-fuse_step:]
    pc_mask = pc_mask_hist[-fuse_step:]

    pc = torch.from_numpy(np.concatenate(pc, axis=0))
    pc_feat = torch.from_numpy(np.concatenate(pc_feat, axis=0))
    pc_mask = torch.from_numpy(np.concatenate(pc_mask, axis=0))

    xyzhe = np.zeros([1,5])
    xyzhe[:,3] = state.heading
    S = torch.FloatTensor([state.location.x, state.location.z, -state.location.y])[None, :] # 1 x 3
    T = torch.from_numpy(transfrom3D(xyzhe))[0, :, :]    # 4 x 4

    pc = pc - S
    pc1 = torch.cat([pc, torch.ones([pc.shape[0], 1])], dim=-1) # N x 4
    pc1 = torch.matmul(pc1, T.transpose(0, 1))
    pc = pc1[:, :3]
    above_mask = (pc[:, 1] > 0.5)

    # discretize point cloud
    vertex_to_map_xz = (pc[:, [0,2]] / MAP_RES).round() + (MAP_DIM-1)//2
    outside_mask = (vertex_to_map_xz[:,0] >= MAP_DIM) +\
                   (vertex_to_map_xz[:,1] >= MAP_DIM) +\
                   (vertex_to_map_xz[:,0] < 0) +\
                   (vertex_to_map_xz[:,1] < 0)
    
    # get y for projection
    y_values = pc[:,1]
    min_y = y_values.min()
    y_values = y_values - min_y
    
    # fliter vertex
    mask = pc_mask | above_mask | outside_mask
    y_values = y_values[~mask]
    vertex_to_map_xz = vertex_to_map_xz[~mask]
    vertex_to_pc_feat = pc_feat[~mask]

    # projection 
    feat_index = (MAP_DIM*vertex_to_map_xz[:,1] + vertex_to_map_xz[:,0]).long()
    flat_highest_y = -torch.ones(int(MAP_DIM * MAP_DIM))
    flat_highest_y, argmax_flat_spatial_map = scatter_max(
        y_values,
        feat_index,
        dim=0,
        out = flat_highest_y,
    )

    # render map
    flat_highest_y = flat_highest_y.numpy()
    argmax_flat_spatial_map = argmax_flat_spatial_map.numpy()
    bev = vertex_to_pc_feat[argmax_flat_spatial_map-1]
    bev[flat_highest_y==-1] = 0
    bev = bev.reshape((MAP_DIM, MAP_DIM, 3)).numpy()

    return bev

def waypoint(ob, bev, pred_vp=None):
    xyzhe = np.zeros([1,5])
    xyzhe[:,3] = -ob['heading']
    S = np.array(ob['position'])[None, :].astype(np.float32)    # 1 x 3
    S = S[:, [0,2,1]] * np.array([1,1,-1])                      # x, z, -y
    T = transfrom3D(xyzhe)[0, :, :]                             # 4 x 4

    cand_pos = np.array([c['position'] for c in ob['candidate']]).astype(np.float32)
    cand_pos = cand_pos[:, [0,2,1]] * np.array([1,1,-1])        # x, z, -y
    cand_pos = cand_pos - S
    ones = np.ones([cand_pos.shape[0], 1]).astype(np.float32)
    cand_pos1 = np.concatenate([cand_pos, ones], axis=-1)
    cand_pos1 = np.dot(cand_pos1, T.transpose(0,1))
    cand_pos = cand_pos1[:, :3]
    cand_pos = (cand_pos[:, [0,2]] / MAP_RES).round() + (MAP_DIM-1)//2
    cand_pos[cand_pos < 0] = 0
    cand_pos[cand_pos >= MAP_DIM] = MAP_DIM - 1
    cand_pos = cand_pos.astype(np.int)

    for i, (p, cand) in enumerate(zip(cand_pos, ob['candidate'])):
        color = (0,0,255) if cand['viewpointId'] == pred_vp  else (0,255,0)
        cv2.putText(bev, str(i), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return bev

def waypoint_state(state, bev):
    xyzhe = np.zeros([1,5])
    xyzhe[:,3] = -state.heading
    S = np.array([state.location.x, state.location.z, -state.location.y])[None, :] # 1 x 3
    T = transfrom3D(xyzhe)[0, :, :]    # 4 x 4

    cands = state.navigableLocations
    cands_pos = []
    for cand in cands:
        cands_pos.append([cand.x, cand.z, -cand.y])
    cands_pos = np.array(cands_pos).astype(np.float32)

    cands_pos = cands_pos - S
    ones = np.ones([cands_pos.shape[0], 1]).astype(np.float32)
    cands_pos1 = np.concatenate([cands_pos, ones], axis=-1)
    cands_pos1 = np.dot(cands_pos1, T.transpose(0,1))
    cands_pos = cands_pos1[:, :3]
    cands_bev = (cands_pos[:, [0,2]] / MAP_RES).round() + (MAP_DIM-1)//2
    cands_bev = cands_bev.astype(np.int)

    for i, p in enumerate(cands_bev):
        cv2.putText(bev, str(i), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), thickness=2)
    return bev

def motion(state, bev, path, graphs):
    xyzhe = np.zeros([1,5])
    xyzhe[:,3] = -state.heading
    S = np.array([state.location.x, state.location.z, -state.location.y])[None, :] # 1 x 3
    T = transfrom3D(xyzhe)[0, :, :]    # 4 x 4

    scan = state.scanId
    pos = []
    for vp in path:
        x,y,z = graphs[scan].nodes[vp]['position'][:3]
        pos.append([x, z, -y])     
    pos = np.array(pos).astype(np.float32)
    bev_pos = pos - S
    ones = np.ones([bev_pos.shape[0], 1]).astype(np.float32)
    bev_pos1 = np.concatenate([bev_pos, ones], axis=-1)
    bev_pos1 = np.dot(bev_pos1, T.transpose(0,1))
    bev_pos = bev_pos1[:, :3]
    bev_pos = (bev_pos[:, [0,2]] / MAP_RES).round() + (MAP_DIM-1)//2
    bev_pos = bev_pos.astype(np.int)

    for s,e in zip(bev_pos[:-1], bev_pos[1:]):
        cv2.line(bev, (s[0],s[1]), (e[0],e[1]), (0,255,0), thickness=2)
    return bev

def draw_instr(txt):
    cap = np.ones([80, 224*4, 3]).astype(np.uint8)*255
    txt_split = []
    split_size = 110
    starts = np.arange(len(txt)//split_size+1) * split_size
    ends = starts + split_size
    for s,e in zip(starts, ends):
        txt_split.append(txt[s:e])
    for i,line in enumerate(txt_split):
        y = 15 + 15 * i
        cv2.putText(cap, line, (5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)
    return cap

def draw_traj(sim, projector, rgb_db, depth_db, graphs, g_paths,
              pred_item, gt_item):

    scan = gt_item['scan']
    instr = gt_item['instruction']
    target = gt_item['path'][-1]
    traj = pred_item['trajectory']
    T = len(traj)
    bev_imgs = []   # output 

    pc_hist = []
    pc_feat_hist = []
    pc_mask_hist = []
    for t, p in enumerate(traj):
        vp, h, e = p[:3]
        e = 0
        if t+1 < T:
            pred_vp = traj[t+1][0]
        else:
            pred_vp = vp
        gt_path = g_paths[scan][vp][target]

        sim.newEpisode([scan], [vp], [h], [e])
        state = sim.getState()[0]
        pc, pc_feat, pc_mask, img = lift(projector, state, rgb_db, depth_db, return_img=True)
        pc_hist.append(pc)
        pc_feat_hist.append(pc_feat)
        pc_mask_hist.append(pc_mask)
        bev = splat(state, pc_hist, pc_feat_hist, pc_mask_hist, fuse_step=2)
        bev = waypoint(state, bev, pred_vp)
        bev = motion(state, bev, gt_path, graphs)
        
        bev = cv2.resize(bev, (224, 224))
        bev_img = np.concatenate([bev, img], axis=1)
        bev_imgs.append(bev_img)

    bev_imgs = np.concatenate(bev_imgs, axis=0)
    # cap = draw_instr(instr)
    # bev_imgs = np.concatenate([bev_imgs, cap], axis=0)
    return bev_imgs

def draw_ob(ob):
    scan = ob['scan']
    vp = ob['viewpoint']
    heading = ob['heading']
    sim.newEpisode([scan], [vp], [heading], [0])
    state = sim.getState()[0]
    bev_imgs = []   # output 

    pc_hist = []
    pc_feat_hist = []
    pc_mask_hist = []
    pc, pc_feat, pc_mask, img = lift(projector, state, rgb_db, depth_db, return_img=True)
    pc_hist.append(pc)
    pc_feat_hist.append(pc_feat)
    pc_mask_hist.append(pc_mask)
    bev = splat(state, pc_hist, pc_feat_hist, pc_mask_hist, fuse_step=1)
    bev = waypoint(ob, bev)
    bev = cv2.resize(bev, (224, 224))
    bev_img = np.concatenate([bev, img], axis=1)
    bev_imgs.append(bev_img)

    bev_imgs = np.concatenate(bev_imgs, axis=0)
    return bev_imgs