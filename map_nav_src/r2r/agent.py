import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import cv2
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.bev_utils import transfrom3D, bevpos_polar, PointCloud
from models.bev_visualize import draw_ob
from models.graph_utils import MAX_DIST, GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _build_projector(self):
        self.bev_dim = self.args.bev_dim
        self.bev_res = self.args.bev_res
        self.projector = PointCloud(math.radians(90),
                                    1,
                                    feature_map_height=14,
                                    feature_map_width=14,
                                    map_dim=self.bev_dim,
                                    map_res=self.bev_res,
                                    world_shift_origin=torch.FloatTensor([0,0,0]).cuda(),
                                    z_clip_threshold=0.5,
                                    device=torch.device('cuda'))

        self.bev_pos = bevpos_polar(self.bev_dim).cuda()
        self.bev_pos = self.bev_pos[None,:,:,:].expand(self.args.batch_size,-1,-1,-1)   # bs x map_dim x map_dim x 3 

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def lift(self, obs):
        '''  unproject rgbs and depths to pointcloud in world coord '''
        bs = len(obs)
        views = 12

        xyzhe = np.zeros([bs, views, 5])
        for i, ob in enumerate(obs):
            x, y, z = ob['position']
            xyzhe[i, :, 0] = x
            xyzhe[i, :, 1] = z
            xyzhe[i, :, 2] = -y
            xyzhe[i, :, 3] = -(np.arange(12) * np.radians(30) + ob['heading']) # counter-clock
            xyzhe[i, :, 4] = np.pi
        T = transfrom3D(xyzhe.reshape(-1,5))
        T = torch.from_numpy(T).cuda()  # bs*views, 4, 4

        depths = np.stack([ob['depth'] for ob in obs], axis=0).reshape(-1, 1, 14, 14)
        depths = torch.from_numpy(depths*10).cuda()
        pc, pc_mask = self.projector.forward(depths, T)
        pc = pc.reshape(bs, -1, 3)
        pc_mask = pc_mask.reshape(bs, -1)
        
        rgbs = np.stack([ob['rgb'] for ob in obs], axis=0).reshape(-1, 14, 14, 768)
        rgbs = torch.from_numpy(rgbs).cuda()
        pc_feat = rgbs.reshape(bs, -1, 768)

        return pc, pc_mask, pc_feat

    def splat(self, obs, pc, pc_mask, pc_feat):
        '''
            1. transform pointcloud to ego coord
            2. project to bev
        '''
        bs = len(obs)

        S = []
        for i, ob in enumerate(obs):
            x, y, z = ob['position']
            S.append([np.array([x, z, -y])])
        S = np.vstack(S).astype(np.float32)              # bs, 3
        S = torch.from_numpy(S).cuda()
        xyzhe = np.zeros([bs, 5])
        for i, ob in enumerate(obs):
            xyzhe[i, 3] = ob['heading']
        T = torch.from_numpy(transfrom3D(xyzhe)).cuda()  # bs, 4, 4

        # transform to ego coord
        pc = pc - S[:, None, :]
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).cuda()
        pc1 = torch.cat([pc, ones], dim=-1)               # bs, N, 4
        pc1 = torch.matmul(pc1, T.transpose(1, 2))        # bs, N, 4
        pc = pc1[:, :, :3]                                # bs, N, 3

        # project to bev
        viz = False
        bev_fts, bev_masks = self.projector.project_bev(pc, pc_mask, pc_feat)
        if viz:
            feat_masks = []
            for ob, bev_mask in zip(obs, bev_masks):
                cand_pos = self._map_cand_to_bev(ob)
                bev_mask = bev_mask.cpu().numpy()[:,:,None] * np.array([255,255,255])[None,None,:]
                bev_mask = bev_mask.astype(np.uint8)
                for p in cand_pos:
                    bev_mask[p[1], p[0], :] = np.array([0,255,0]).astype(np.uint8)
                feat_masks.append(bev_mask)
            feat_masks = np.concatenate(feat_masks, axis=1)
            cv2.imwrite('feat_masks.png', feat_masks)

            bev_imgs = [draw_ob(ob) for ob in obs]
            bev_imgs = np.concatenate(bev_imgs, axis=0)
            cv2.imwrite('bev_imgs.png', bev_imgs)

        bev_masks = torch.ones_like(bev_masks)
        bev_fts = bev_fts.reshape(bs, -1, 768)
        bev_masks = bev_masks.reshape(bs, -1)
        bev_pos_fts = self.bev_pos.reshape(bs, -1, 3)

        return bev_fts, bev_masks, bev_pos_fts

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda, zero is stop token

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j]) / MAX_DIST

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _map_cand_to_bev(self, ob):
        bev_dim = self.bev_dim
        bev_res = self.bev_res

        S = np.array(ob['position'])[None, :].astype(np.float32)                      # 1 x 3
        S = S[:, [0,2,1]] * np.array([1,1,-1], dtype=np.float32)                      # x, z, -y
        xyzhe = np.zeros([1,5])
        xyzhe[:,3] = -ob['heading']
        T = transfrom3D(xyzhe)[0, :, :]                                               # 4 x 4

        cand_pos = np.array([c['position'] for c in ob['candidate']]).astype(np.float32)
        cand_pos = cand_pos[:, [0,2,1]] * np.array([1,1,-1], dtype=np.float32)        # x, z, -y
        cand_pos = cand_pos - S
        ones = np.ones([cand_pos.shape[0], 1]).astype(np.float32)
        cand_pos1 = np.concatenate([cand_pos, ones], axis=-1)
        cand_pos1 = np.dot(cand_pos1, T.transpose(0,1))
        cand_pos = cand_pos1[:, :3]
        cand_pos = (cand_pos[:, [0,2]] / bev_res).round() + (bev_dim-1)//2
        cand_pos[cand_pos<0] = 0
        cand_pos[cand_pos>=bev_dim] = bev_dim - 1
        cand_pos = cand_pos.astype(np.int)

        return cand_pos

    def _nav_bev_variable(self, obs, gmaps):
        batch_pc = []
        batch_pc_mask = []
        batch_pc_feat = []

        batch_bev_nav_masks = []
        batch_bev_cand_vpids = []
        batch_bev_cand_idxs = []
        batch_gmap_pos_fts = []
        for i, (ob, gmap) in enumerate(zip(obs, gmaps)):
            vp = ob['viewpoint']
            pc, pc_mask, pc_feat = gmap.gather_node_pc(vp, self.args.pc_order)
            batch_pc.append(pc)
            batch_pc_mask.append(pc_mask)
            batch_pc_feat.append(pc_feat)
            # map candidate to bev
            bev_cand_vpids = [None] + [c['viewpointId'] for c in ob['candidate']]       # stop is None
            bev_cand_pos = self._map_cand_to_bev(ob)
            bev_cand_idxs = bev_cand_pos[:,1] * self.bev_dim + bev_cand_pos[:,0]
            bev_cand_idxs = np.insert(bev_cand_idxs, 0, (self.bev_dim*self.bev_dim-1)//2) # stop token, the center of BEV
            bev_nav_masks = np.zeros(self.bev_dim * self.bev_dim).astype(np.bool)
            bev_nav_masks[bev_cand_idxs] = True
            batch_bev_cand_vpids.append(bev_cand_vpids)
            batch_bev_nav_masks.append(torch.from_numpy(bev_nav_masks))
            batch_bev_cand_idxs.append(torch.from_numpy(bev_cand_idxs))
            # global relative pos
            gmap_pos_fts = gmap.get_pos_fts(ob['viewpoint'], [gmap.start_vp], ob['heading'], ob['elevation'])
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
        
        # collate
        batch_pc = pad_sequence(batch_pc, batch_first=True)
        batch_pc_mask = pad_sequence(batch_pc_mask, batch_first=True, padding_value=True)   # no depth mask
        batch_pc_feat = pad_sequence(batch_pc_feat, batch_first=True)
        batch_bev_nav_masks = pad_tensors(batch_bev_nav_masks).cuda()
        batch_bev_cand_idxs = pad_tensors(batch_bev_cand_idxs)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).expand(-1, self.bev_dim*self.bev_dim, -1).cuda()

        bev_fts, bev_masks, bev_pos_fts = self.splat(obs, batch_pc, batch_pc_mask, batch_pc_feat)
        bev_pos_fts = torch.cat([batch_gmap_pos_fts, bev_pos_fts], dim=-1)

        return {
            'bev_fts': bev_fts, 'bev_pos_fts': bev_pos_fts,
            'bev_masks': bev_masks, 'bev_nav_masks': batch_bev_nav_masks,
            'bev_cand_idxs': batch_bev_cand_idxs, 'bev_cand_vpids': batch_bev_cand_vpids,
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = 0 # (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # pointcloud representation
            pc, pc_mask, pc_feat = self.lift(obs)

            # add to gmap
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    gmap.update_node_pc(i_vp, pc[i], pc_mask[i], pc_feat[i])
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(self._nav_bev_variable(obs, gmaps))
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['bev_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    # nav_targets = self._teacher_action(
                    #     obs, nav_vpids, ended, 
                    #     visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                    # )
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                else:
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                                                 
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj
