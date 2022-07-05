#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# modified from https://github.com/facebookresearch/DepthContrast/blob/main/criterions/nce_loss_moco.py

import torch
from torch import nn
from torch import Tensor
from typing import List
from loss import _gather

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not (torch.distributed.is_initialized()):
        return tensor
    
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def global_features(x:Tensor, pool="max") -> Tensor:
    """aggregate global features.

    Args:
        x (Tensor): a feature map or pointwise feature

    Returns:
        Tensor: (B,C)
    """
    l = x.dim()
    if l == 3:
        # point features
        if pool == "max":
            y = nn.functional.adaptive_max_pool1d(x, 1, False)
        elif pool == "average":
            y = nn.functional.adaptive_avg_pool1d(x, 1)
        else:
            raise ValueError("only support max and average pooling")
        y = torch.squeeze(y, 2)
    elif l==4:
        # feature map
        if pool == "max":
            y = nn.functional.adaptive_max_pool2d(x, (1, 1), False)
        elif pool == "average":
            y = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        else:
            raise ValueError("only support max and average pooling")
        y = y.squeeze(-1).squeeze(-1)
    else:
        raise ValueError("only support 3 or 4 dims, but get {:d}".format(l))    
    return y


class NCELossMoco(nn.Module):
    def __init__(self,
                num_negative:int,
                dim:int,
                temp:float,
                weight:List[float]=[0.5, 0.5],                 
                normalize_embedding:List[bool]=[False, False]):
        super(NCELossMoco, self).__init__()
        self.w12 = weight[0]    # pos-neg: format 1-2
        self.w21 = weight[1]    # format 2-1
        self.K = num_negative
        self.dim = dim
        self.T = temp
        # buffer for format 1
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # buffer for format 2
        self.register_buffer("queue_other", torch.randn(self.dim, self.K))
        self.queue_other = nn.functional.normalize(self.queue_other, dim=0)
        self.register_buffer("queue_other_ptr", torch.zeros(1, dtype=torch.long))
        # cross-entropy loss.
        self.xe_criterion = nn.CrossEntropyLoss()
        
        self.normalize_embedding = normalize_embedding

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, okeys=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        self.queue_ptr[0] = ptr

        # gather keys before updating queue
        okeys = concat_all_gather(okeys)
            
        other_ptr = int(self.queue_other_ptr)
    
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_other[:, other_ptr:other_ptr + batch_size] = okeys.T
        other_ptr = (other_ptr + batch_size) % self.K  # move pointer
    
        self.queue_other_ptr[0] = other_ptr
        
    def forward(self, x1:Tensor, x2:Tensor):
        """
        Args:
            x1 (Tensor): (N, C). global features
            x2 (Tensor): (N, C). global features
        """
        if self.normalize_embedding[0]:
            x1 = nn.functional.normalize(x1, dim=1, p=2)
        if self.normalize_embedding[1]:
            x2 = nn.functional.normalize(x2, dim=1, p=2)
        
        # contrast: 1-2
        l_pos12 = torch.einsum("nc,nc->n", [x1, x2]).unsqueeze(-1)
        l_neg12 = torch.einsum("nc,ck->nk", [x1, self.queue_other.clone().detach()])
        logits12 = torch.cat([l_pos12, l_neg12], dim=1)     # (N, K+1)
        logits12 /= self.T
        
        # contrast: 2-1
        l_neg21 = torch.einsum("nc,ck->nk", [x2, self.queue.clone().detach()])
        logits21 = torch.cat([l_pos12, l_neg21], dim=1)     # (N, K+1)
        logits21 /= self.T
        
        # save buffer
        self._dequeue_and_enqueue(x1, x2)

        labels = torch.zeros(
            logits12.shape[0], device=logits12.device, dtype=torch.int64
        )
        
        loss12 = self.xe_criterion(logits12, labels)
        loss21 = self.xe_criterion(logits21, labels)
        
        return loss12 * self.w12 + loss21 * self.w21


class NCELossMocoV2(nn.Module):
    def __init__(self,
                num_negative:int,
                dim:int,
                temp:float,                
                normalize_embedding:List[bool]=[False, False]):
        super(NCELossMocoV2, self).__init__()
        self.K = num_negative
        self.dim = dim
        self.T = temp
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # cross-entropy loss.
        self.xe_criterion = nn.CrossEntropyLoss()
        self.normalize_embedding = normalize_embedding

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
        
    def forward(self, x1:Tensor, x2:Tensor):
        """
        Args:
            x1 (Tensor): (N, C). global features
            x2 (Tensor): (N, C). global features
        """
        if self.normalize_embedding[0]:
            x1 = nn.functional.normalize(x1, dim=1, p=2)
        if self.normalize_embedding[1]:
            x2 = nn.functional.normalize(x2, dim=1, p=2)
        
        # contrast: 1-2
        l_pos12 = torch.einsum("nc,nc->n", [x1, x2]).unsqueeze(-1)
        l_neg12 = torch.einsum("nc,ck->nk", [x1, self.queue.clone().detach()])
        logits12 = torch.cat([l_pos12, l_neg12], dim=1)     # (N, K+1)
        logits12 /= self.T
        
        # save buffer
        self._dequeue_and_enqueue(x2)

        labels = torch.zeros(
            logits12.shape[0], device=logits12.device, dtype=torch.int64
        )

        loss12 = self.xe_criterion(logits12, labels)
        
        return loss12


@torch.no_grad()
def momentum_update(model, model_m, m=0.99):
    for p, pm in zip(model.parameters(), model_m.parameters()):
        pm.data = pm.data * m + p.data * (1 - m)


@torch.no_grad()
def batch_shuffle_ddp(x, idx=None, return_idx_shuffle=False):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    if idx is None:
        idx_shuffle = torch.randperm(batch_size_all).cuda()
    else:
        # share shuffle index for multiple tensors
        idx_shuffle = idx

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    if return_idx_shuffle:
        return x_gather[idx_this], idx_unshuffle, idx_shuffle
    else:
        return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


def build_mlp_head(dim_model, dim_inter, dim_out):
    return nn.Sequential(
        nn.Linear(dim_model, dim_inter),
        nn.ReLU(True),
        nn.Linear(dim_inter, dim_out)
    )

def build_dense_head(dim_model, dim_inter, dim_out):
    return nn.Sequential(
        nn.Conv1d(dim_model, dim_inter, 1),
        nn.ReLU(True),
        nn.Conv1d(dim_inter, dim_out, 1)
    )


class UNetHead(nn.Module):
    def __init__(self, encoder, dim_model, dim_inter, dim_out, dense_head=False):
        super().__init__()
        self.encoder = encoder
        self.mlp = build_mlp_head(dim_model, dim_inter, dim_out)
        self.dense_head = dense_head
        if dense_head:
            self.dh = build_dense_head(dim_model, dim_inter, dim_out)
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor, return_fm=False):
        fm = self.encoder(x)
        B, C, _, _ = fm.size()
        fm_lin = fm.view(B, C, -1)
        fm_masked = _gather(fm_lin, mask)
        y = global_features(fm_masked)
        y = self.mlp(y)        
        if self.dense_head:
            fm_masked = self.dh(fm_masked)
        if return_fm:
            return y, fm, fm_masked
        else:
            return y
        

class PointNet2Head(nn.Module):
    def __init__(self, encoder, dim_model, dim_inter, dim_out, dense_head=False):
        super().__init__()
        self.encoder = encoder
        self.mlp = build_mlp_head(dim_model, dim_inter, dim_out)
        self.dense_head = dense_head
        if dense_head:
            self.dh = build_dense_head(dim_model, dim_inter, dim_out)
    
    def forward(self, x:torch.Tensor, fps_ind=None, return_fm=False):
        if fps_ind is None:
            ret = self.encoder(x)
        else:
            ret = self.encoder(x, fps_ind)
        feats = ret["fp2_features"]
        y = global_features(feats)
        y = self.mlp(y)
        if self.dense_head:
            feats = self.dh(feats)        
        if return_fm:
            return y, feats
        else:
            return y    


if __name__ == "__main__":
    net = NCELossMoco(2048, 256, 0.01)
    for i in range(10000):
        x1 = torch.rand(8, 256, 28, 28)
        x2 = torch.rand(8, 256, 30)
        x1 = global_features(x1)
        x2 = global_features(x2)
        loss = net(x1, x2)
        if i%100==0:
            print("[{:05d}] Loss: {:.5f}  Pointer: {:d} {:d}"
                  .format(i, loss.item(), int(net.queue_ptr), int(net.queue_other_ptr)))    
