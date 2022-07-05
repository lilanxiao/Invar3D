# ------------------------------------------------
import cv2      # first import cv2, then torch
cv2.setNumThreads(0)
# https://github.com/pytorch/pytorch/issues/1838
# no need to use OpenCV with multithreading since 
# Pytorch alread uses multiprocessing 
# ------------------------------------------------
import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from loss import point_info_nce_loss, _gather
from utils.sparse_helper import collate_dense_sparse, to_device, align_sparse_features
from loss_moco import (build_dense_head, build_mlp_head, NCELossMocoV2, momentum_update, 
                       batch_shuffle_ddp, batch_unshuffle_ddp)
from scannet.scannet_pretrain import ScanNetDepthVoxelDataset
from model.cnn3d.minkunet import MinkUNet34C
from model.cnn2d.conv2d import UNet, BasicBlock
from meters import ProgressMeter


class UNetHead(nn.Module):
    def __init__(
        self, 
        fusion,
        feature_dims=[64, 64],
        dim_model : int = 128,
        dim_inter : int = 512,
        dim_out : int = 128
        ):
        super().__init__()
        self.fusion = fusion
        self.encoder = UNet(fusion=fusion)
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(),
                                        BasicBlock(128, 128))
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(128+feature_dims[1], 128, kernel_size=2, stride=2, padding=0, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(),
                                        BasicBlock(128, 128))
        self.conv = BasicBlock(128+feature_dims[0], 128)
        self.dense_head = build_dense_head(dim_model, dim_inter, dim_out)
        self.ghead = build_mlp_head(dim_model, dim_inter, dim_out)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, depth : Tensor, ind : Tensor, rgb : Tensor = None):
        if self.fusion:
            input = {"depthmap":depth, "rgb":rgb}
        else:
            input = depth
        # s_y=8, s_x1=4, s_x2=2
        y, x1, x2 = self.encoder(input, True)
        y = self.upsample1(y)   # s=4
        y = torch.cat([y, x2], dim=1)
        y = self.upsample2(y)   # s=2
        y = torch.cat([y, x1], dim=1)
        y = self.conv(y)        # s=2, B,C,H,W
        B, C, _, _ = y.size()
        y_lin = y.view(B, C, -1)
        feats = _gather(y_lin, ind)
        local_feats = self.dense_head(feats)
        local_feats = torch.nn.functional.normalize(local_feats, dim=1)
        global_feats = self.pool(y).view(B, C)
        global_feats = self.ghead(global_feats)
        global_feats = torch.nn.functional.normalize(global_feats, dim=1)
        return global_feats, local_feats


class SparseUNetHead(nn.Module):
    def __init__(
        self, 
        rgb : bool = True,
        dim_model : int = 256,
        dim_inter : int = 512,
        dim_out : int = 128
        ):
        super().__init__()
        dim_in = 3 if rgb else 1
        self.encoder = MinkUNet34C(dim_in, dim_model)
        self.dense_head = build_dense_head(dim_model, dim_inter, dim_out)
        self.ghead = build_mlp_head(dim_model, dim_inter, dim_out)
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, sparse_tensor : ME.SparseTensor, index : Tensor) :
        sout = self.encoder(sparse_tensor)
        # BxCxN
        feats = align_sparse_features(sout, index)
        local_feats = self.dense_head(feats)
        local_feats = torch.nn.functional.normalize(local_feats, dim=1)
        global_feats = self.pool(feats).squeeze(2)
        global_feats = self.ghead(global_feats)
        global_feats = torch.nn.functional.normalize(global_feats, dim=1)
        return global_feats, local_feats


class DepthVoxelContrast(nn.Module):
    def __init__(self,
                 dim_out,
                 dim_inter,
                 num_neg,
                 temperature,
                 momentum=0.999,
                 fusion=True,
                 local_loss=True,
                 global_loss=True,
                 within_format=False,
                 ddp=True,
                 warmup=-1):
        super().__init__()
        self.dim_out = dim_out
        self.dim_inter = dim_inter
        self.num_neg = num_neg
        self.t = temperature
        self.momentum = momentum
        assert (local_loss or global_loss), "at least one of local and global loss have to be true"
        self.local_loss = local_loss
        self.global_loss = global_loss
        self.within_format = within_format
        self.fusion = fusion
        self.ddp = ddp
        if self.global_loss and self.local_loss:
            self.warmup = warmup
        else:
            self.warmup = -1
        # initialize models
        self.net2d = UNetHead(fusion, dim_inter=dim_inter, dim_out=dim_out)
        self.net3d = SparseUNetHead(fusion, dim_inter=dim_inter, dim_out=dim_out)

        if self.global_loss:
            self.net2d_m = UNetHead(fusion, dim_inter=dim_inter, dim_out=dim_out)
            self._copy(self.net2d, self.net2d_m)
            self.net3d_m = SparseUNetHead(fusion, dim_inter=dim_inter, dim_out=dim_out)
            self._copy(self.net3d, self.net3d_m)
            self.loss_func1 = NCELossMocoV2(num_neg, dim_out, temperature)
            self.loss_func2 = NCELossMocoV2(num_neg, dim_out, temperature)
            if within_format:
                self.loss_func3 = NCELossMocoV2(num_neg, dim_out, temperature)
                self.loss_func4 = NCELossMocoV2(num_neg, dim_out, temperature)

    def _copy(self, model:nn.Module, model_m:nn.Module):
        for p, pm in zip(model.parameters(), model_m.parameters()):
            pm.data.copy_(p.data)
            pm.requires_grad = False
    
    def update_momentum_models(self):
        momentum_update(self.net2d, self.net2d_m, self.momentum)
        momentum_update(self.net3d, self.net3d_m, self.momentum)

    def forward(self, data : Dict[str, Tensor], epoch=None):
        q2d, f2d = self.net2d(
            depth = data["depthmap1"],
            rgb = data["rgb1"] if self.fusion else None,
            ind = data["ind_depthmap1"]
        )
        q3d, f3d = self.net3d(data["sin1"], data["ind_vox1"])
        if self.local_loss:
            loss_pc = point_info_nce_loss(f2d, f3d, self.t)["loss"]
        else:
            loss_pc = torch.tensor(0.).to(q2d.device)
        
        if self.global_loss:
            with torch.no_grad():
                dm = data["depthmap2"]
                rgb = data["rgb2"]
                idm = data["ind_depthmap2"]
                sin = data["sin2"]
                iv = data["ind_vox2"]
                if self.ddp:
                    # BN shuffle
                    # NOTE: no need for Sparse CNN, according to DepthContrast code
                    dm, idx_unshuffle, idx_shuffle = batch_shuffle_ddp(dm, return_idx_shuffle=True)
                    rgb, _ = batch_shuffle_ddp(rgb, idx=idx_shuffle)
                    idm, _ = batch_shuffle_ddp(idm, idx=idx_shuffle)
                k2d, _ = self.net2d_m(depth=dm, rgb=rgb, ind=idm)
                k3d, _ = self.net3d_m(sin, iv)
                if self.ddp:
                    # unshuffle
                    k2d = batch_unshuffle_ddp(k2d, idx_unshuffle)
            
            # 3d model generates keys, 2d model generates queries
            loss23 = self.loss_func1(q2d, k3d)
            # 3d model generates queries, 2d model generates keys
            loss32 = self.loss_func2(q3d, k2d)
            if self.within_format:
                # consider within format
                loss22 = self.loss_func3(q2d, k2d)
                loss33 = self.loss_func4(q3d, k3d)
                loss_g = (loss23 + loss32 + loss22 + loss33)/4.
            else:
                # only consider cross-format
                loss_g = (loss23 + loss32)/2
            
            if self.warmup > 0 and epoch is not None:
                wl = 0.5 * min(epoch/self.warmup, 1)
                wg = 1 - wl
                loss = loss_g * wg + loss_pc * wl
            else:
                loss = (loss_pc + loss_g)/(float(self.local_loss)+float(self.global_loss)) 
            
            metric = {"loss32": loss32.item(), 
                    "loss23": loss23.item(),
                    "loss_g": loss_g.item(),
                    "loss_pc": loss_pc.item(),
                    "loss": loss.item()}
            if self.within_format:
                metric["loss22"] = loss22.item()
                metric["loss33"] = loss33.item()
        else:
            # only PointInfoNCE Loss
            loss = loss_pc
            metric = {"loss_pc": loss_pc.item(),
                      "loss": loss.item()}
        return loss, metric   


def main():
    # hyperparamterts
    BATCH_SIZE = 8
    EPOCH = 50
    TEMPERATURE = 0.07
    LOCAL_LOSS = True
    GLOBAL_LOSS = True
    NUM_MATCH = 1024
    SPARSE_KEY = ["coords", "feats"]
    DEVICE = "cuda:0"
    VOXEL_SIZE = 0.05
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    SAVE = os.path.join(BASE_DIR, "DVCo_test")
    WARMUP = 10
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)

    net = DepthVoxelContrast(dim_out=128, dim_inter=512, num_neg=4096*8,
                             ddp=False, local_loss=LOCAL_LOSS, global_loss=GLOBAL_LOSS,
                             temperature=TEMPERATURE, warmup=WARMUP)
    net.to(DEVICE)
    ds = ScanNetDepthVoxelDataset(
        "train", num_match=NUM_MATCH, match_thresh=0.05, voxel_size=VOXEL_SIZE, num_pairs=2)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    drop_last=True, collate_fn=collate_dense_sparse(SPARSE_KEY, 2), num_workers=6)
    
    val_ds = ScanNetDepthVoxelDataset(
        "val", num_match=NUM_MATCH, match_thresh=0.05, voxel_size=VOXEL_SIZE, num_pairs=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True,
                    drop_last=True, collate_fn=collate_dense_sparse(SPARSE_KEY, 2), num_workers=6) 
    
    optimizer = optim.SGD(net.parameters(), lr=0.03,
                          weight_decay=1e-4, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)
    for epoch in range(EPOCH):
        net.train()
        progress = ProgressMeter(
                    len(dl),
                    prefix="Epoch: [{}]".format(epoch))
        
        for i, data in enumerate(dl):
            to_device(data, DEVICE)
            data["sin1"] = ME.SparseTensor(
                data["feats1"], data["coords1"]
            )
            data["sin2"] = ME.SparseTensor(
                data["feats2"], data["coords2"]
            )
            
            loss, metric = net(data, epoch+1)
            progress.update(i, metric)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if GLOBAL_LOSS:
                net.update_momentum_models()
            
            if i % 20 == 0:
                progress.display(i)
                torch.cuda.empty_cache()
        scheduler.step()
        
        torch.save({
            "epoch": epoch, 
            "model_state_dict": net.net2d.encoder.state_dict()},
            os.path.join(SAVE, "ckpt_{:03d}_depth.pth".format(epoch)))
        torch.save({
            "epoch": epoch, 
            "model_state_dict": net.net3d.encoder.state_dict()},
            os.path.join(SAVE, "ckpt_{:03d}_pcd.pth".format(epoch)))
        
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                # do nothing. hack to avoid OOM caused by MinkowskiEngine
                to_device(data, DEVICE)
                data["sin1"] = ME.SparseTensor(
                    data["feats1"], data["coords1"]
                )
                data["sin2"] = ME.SparseTensor(
                    data["feats2"], data["coords2"]
                )
                out = net.net3d(data["sin1"], data["ind_vox1"])
                
                if i > 20:
                    break
            

if __name__ == "__main__":
    main()
    
