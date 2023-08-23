from collections import defaultdict
import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .ops import pad_tensors_wgrad, gen_seq_masks
from .bev_utils import bevpos_polar, PointCloud

BEV_DIM = 21
BEV_RES = 0.5

def build_projector():
    projector = PointCloud(math.radians(90),
                           1,
                           feature_map_height=14,
                           feature_map_width=14,
                           map_dim=BEV_DIM,
                           map_res=BEV_RES,
                           world_shift_origin=torch.FloatTensor([0,0,0]).cuda(),
                           z_clip_threshold=0.5,
                           device=torch.device('cuda'))

    bev_pos = bevpos_polar(BEV_DIM).cuda()
    bev_pos = bev_pos.reshape(BEV_DIM * BEV_DIM, 3)[None, :, :] # 1 x 441 x 3
    return projector, bev_pos

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class MulClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 40))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)
        self.drop_env = nn.Dropout(config.feat_dropout)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            else:
                self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)
        if 'sem' in config.pretrain_tasks or 'masksem' in config.pretrain_tasks:
            self.local_sem_head = MulClsPrediction(self.config.hidden_size)
            self.sem_pred_token = self.config.sem_pred_token

        self.init_weights()
        self.tie_weights()
        self.projector, self.bev_pos_fts = build_projector()

    def drop_feats(self, batch):
        for ft_name in ['traj_view_img_fts', 'traj_obj_img_fts', 'bev_fts']:
            if ft_name in batch:
                batch[ft_name] = self.drop_env(batch[ft_name])
        return batch
        

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def lift_splat(self, batch):
        rgbs = batch.pop('rgbs')
        depths = batch.pop('depths')
        sems = batch.pop('sems')
        T_c2w = batch.pop('T_c2w')
        T_w2c = batch.pop('T_w2c')
        S_w2c = batch.pop('S_w2c')
        bev_gpos_fts = batch.pop('bev_gpos_fts')
        bs = rgbs.shape[0]

        # lift
        depths_var = (depths * 10).reshape(-1, 1, 14, 14)   # bs*views, 1, 14, 14
        pc, pc_mask = self.projector.forward(depths_var, T_c2w.reshape(-1, 4, 4))
        pc = pc.reshape(bs, -1, 3)          # B, N, 3
        pc_mask = pc_mask.reshape(bs, -1)   # B, N
        pc_feat = rgbs.reshape(bs, -1, 768) # B, N, 768
        pc_sem = sems.reshape(bs, -1, 40)   # B, N, 40

        # splat
        pc = pc - S_w2c
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).cuda()
        pc1 = torch.cat([pc, ones], dim=-1)                         # bs, N, 4
        pc1 = torch.matmul(pc1, T_w2c.squeeze(1).transpose(1, 2))   # bs, N, 4
        pc = pc1[:, :, :3]                                          # bs, N, 3

        viz = False
        bev_fts, bev_masks, bev_sems, bev_sem_masks = self.projector.project_bev(pc, pc_mask, pc_feat, pc_sem)
        if viz:
            feat_masks = []
            for bev_mask, bev_nav_mask in zip(bev_masks, batch['bev_nav_masks']):
                bev_mask = bev_mask.cpu().numpy()[:,:,None] * np.array([255,255,255])[None,None,:]
                bev_mask = bev_mask.astype(np.uint8)
                bev_nav_mask = bev_nav_mask.reshape(BEV_DIM, BEV_DIM).cpu().numpy()
                bev_mask[bev_nav_mask] = np.array([0,255,0]).astype(np.uint8)
                feat_masks.append(bev_mask)
            feat_masks = np.concatenate(feat_masks, axis=1)
            cv2.imwrite('tmp.png', feat_masks)

        bev_masks = torch.ones_like(bev_masks)
        bev_fts = bev_fts.reshape(bs, -1, 768)
        bev_masks = bev_masks.reshape(bs, -1)
        bev_pos_fts = self.bev_pos_fts.expand(bs, -1, -1)
        bev_pos_fts = torch.cat([bev_gpos_fts.expand(-1, BEV_DIM*BEV_DIM, -1), bev_pos_fts], dim=-1)
        bev_sems = bev_sems.reshape(bs, -1, 40)
        bev_sem_masks = bev_sem_masks.reshape(bs, -1)
        batch.update({
            'bev_fts': bev_fts,
            'bev_masks': bev_masks,
            'bev_pos_fts': bev_pos_fts,
            'bev_sems': bev_sems,
            'bev_sem_masks': bev_sem_masks
        })

        return batch

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'], 
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('mrc'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],  
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'],
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],  
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'], batch['gmap_visited_masks'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'], batch['bev_cand_idxs'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss
            )
        elif task.startswith('og'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],  
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('sem'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_sem(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],  
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'],
                batch['bev_sems'], batch['bev_sem_masks'], compute_loss
            )
        elif task.startswith('masksem'):
            batch = self.lift_splat(batch)
            batch = self.drop_feats(batch)
            return self.forward_masksem(
                batch['txt_ids'], batch['txt_lens'], 
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],  
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_pair_dists'], batch['gmap_vpids'],
                batch['bev_fts'], batch['bev_pos_fts'], batch['bev_masks'], batch['bev_nav_masks'],
                batch['bev_sems'], batch['bev_sem_masks'], batch['bev_mrc_masks'], compute_loss
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
        txt_labels, compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrc(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
        vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        gmap_embeds, bev_embeds, obj_embeds, obj_masks = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
            return_gmap_embeds=False
        )
        
        obj_masked_output = self._compute_masked_hidden(obj_embeds, vp_obj_mrc_masks)
        obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
        obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)

        if compute_loss:
            obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
            obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
            mrc_loss = obj_mrc_loss
            return mrc_loss
        else:
            return obj_prediction_soft_labels, obj_mrc_targets

    def index_cand_embeds(self, nav_masks, embeds, cand_idxs):
        bs, max_len = cand_idxs.size(0), cand_idxs.size(1)
        batch_idxs = torch.arange(bs)[:,None].repeat(1, max_len).to(cand_idxs.device)
        cand_embeds = embeds[batch_idxs, cand_idxs]
        cand_masks = nav_masks[batch_idxs, cand_idxs]
        return cand_embeds, cand_masks

    def forward_sap(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, gmap_visited_masks,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks, bev_cand_idxs,
        global_act_labels, local_act_labels, compute_loss
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, bev_embeds, obj_embeds, obj_masks = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            CENTER = (self.config.bev_dim*self.config.bev_dim-1) // 2
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], bev_embeds[:, CENTER]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        cand_embeds, cand_masks = self.index_cand_embeds(bev_nav_masks, bev_embeds, bev_cand_idxs)
        local_logits = self.local_sap_head(cand_embeds).squeeze(2) * (1 - fuse_weights)
        local_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate([None] + traj_cand_vpids[i][-1]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
        obj_labels, compute_loss
    ):
        gmap_embeds, bev_embeds, obj_embeds, obj_masks = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
            return_gmap_embeds=False
        )

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sem(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
        bev_sems, bev_sem_masks, compute_loss
    ):

        bev_embeds = self.bert.forward_sem(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks, sem_pred_token=self.sem_pred_token
        )
        
        masked_output = self._compute_masked_hidden(bev_embeds, bev_sem_masks)
        sem_logits = self.local_sem_head(masked_output)
        sem_labels = bev_sems[bev_sem_masks].float()

        if compute_loss:
            sem_loss = F.binary_cross_entropy_with_logits(sem_logits, sem_labels, reduction='none')
            return sem_loss
        else:
            return sem_logits, sem_labels

    def forward_masksem(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
        bev_fts, bev_pos_fts, bev_masks, bev_nav_masks,
        bev_sems, bev_sem_masks, bev_mrc_masks, compute_loss
    ):
        bev_mrc_masks_ext = bev_mrc_masks.unsqueeze(-1).expand_as(bev_fts)
        bev_fts = bev_fts.data.masked_fill(bev_mrc_masks_ext, 0)

        bev_embeds = self.bert.forward_sem(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids,
            bev_fts, bev_pos_fts, bev_masks, bev_nav_masks, sem_pred_token=self.sem_pred_token
        )
        
        masked_output = self._compute_masked_hidden(bev_embeds, bev_sem_masks & bev_mrc_masks)
        sem_logits = self.local_sem_head(masked_output)
        sem_labels = bev_sems[bev_sem_masks & bev_mrc_masks].float()

        if compute_loss:
            sem_loss = F.binary_cross_entropy_with_logits(sem_logits, sem_labels, reduction='none')
            return sem_loss
        else:
            return sem_logits, sem_labels