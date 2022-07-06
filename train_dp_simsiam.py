# ------------------------------------------------
import cv2      # first import cv2, then torch
cv2.setNumThreads(0)
# https://github.com/pytorch/pytorch/issues/1838
# ------------------------------------------------
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from itertools import chain

from loss_moco import global_features, momentum_update
from scannet.scannet_pretrain import ScannetDepthPointDataset
from model.cnn2d.conv2d import UNet
from model.pointnet2.backbone_module import Pointnet2Backbone
from meters import ProgressMeter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SimSiam(nn.Module):
    def __init__(self, 
                encoder_d=UNet(), 
                encoder_p=Pointnet2Backbone(), 
                dim=1024, 
                pred_dim=256,
                momentum_encoder=False,
                momentum=0.999):
        super(SimSiam, self).__init__()
        self.encoder_d = encoder_d
        self.encoder_p = encoder_p
        
        self.projector_d = nn.Sequential(
            nn.Linear(pred_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )
        
        self.projector_p = nn.Sequential(
            nn.Linear(pred_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )
        
        self.predictor_d = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, dim)
        )
        
        self.predictor_p = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(True),
            nn.Linear(pred_dim, dim)
        )
        
        self.momentum_encoder = momentum_encoder
        if momentum_encoder:
            self.m_encoder_d = copy.deepcopy(encoder_d)
            self.m_encoder_p = copy.deepcopy(encoder_p)
            self.m_projector_d = copy.deepcopy(self.projector_d)
            self.m_projector_p = copy.deepcopy(self.projector_p)
            self.m_encoder_d.requires_grad_(False)
            self.m_encoder_p.requires_grad_(False)
            self.m_projector_d.requires_grad_(False)
            self.m_projector_p.requires_grad_(False)
        self.m = momentum
        
        self.criterion = nn.CosineSimilarity(1)
    
    def sim_siam_forward(self, depth, points):
        zd = self.encoder_d(depth)
        zd = global_features(zd)
        zd = self.projector_d(zd)
        
        zp = self.encoder_p(points)
        zp = zp["fp2_features"]
        zp = global_features(zp)
        zp = self.projector_p(zp)
        
        pd = self.predictor_d(zd)
        pp = self.predictor_p(zp)

        # only cross format
        loss = self.criterion(pd, zp.detach()).mean() + self.criterion(pp, zd.detach()).mean()
        loss = loss * 0.5        
        return loss

    def byol_forward(self, depth, points):
        zd = self.encoder_d(depth)
        zd = global_features(zd)
        zd = self.projector_d(zd)
        
        zp = self.encoder_p(points)
        zp = zp["fp2_features"]
        zp = global_features(zp)
        zp = self.projector_p(zp)
        
        pd = self.predictor_d(zd)
        pp = self.predictor_p(zp)
        
        with torch.no_grad():
            mzd = self.m_encoder_d(depth)
            mzd = global_features(mzd)
            mzd = self.m_projector_d(mzd)
            
            mzp = self.m_encoder_p(points)
            mzp = mzp["fp2_features"]
            mzp = global_features(mzp)
            mzp = self.m_projector_p(mzp)
        
        loss = self.criterion(pd, mzp).mean() + self.criterion(pp, mzd).mean()
        loss *= 0.5
        
        momentum_update(self.encoder_d, self.m_encoder_d, self.m)
        momentum_update(self.encoder_p, self.m_encoder_p, self.m)
        momentum_update(self.projector_d, self.m_projector_d, self.m)
        momentum_update(self.projector_p, self.m_projector_p, self.m)
        
        return loss 
            
    def forward(self, depth, points):
        if self.momentum_encoder:
            return self.byol_forward(depth, points)
        else:
            return self.sim_siam_forward(depth, points)



def main():
    # hyperparamterts
    BATCH_SIZE = 6
    EPOCH = 100
    DEVICE = "cuda:0"
    LR = 0.05 / 256 * BATCH_SIZE
    MOMENTUM_ENCODER = True
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    SAVE = os.path.join(BASE_DIR, "DP_SimSiam")
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    
    ds = ScannetDepthPointDataset("train", diff_crop=True, augment=True, num_match=10) # smalle num_match for speed
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    
    encoder_d = UNet()
    encoder_p = Pointnet2Backbone()
    net = SimSiam(encoder_d, encoder_p, dim=1024, pred_dim=256, momentum_encoder=MOMENTUM_ENCODER).to(DEVICE)

    optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=1e-4, 
                          momentum=0.9)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, EPOCH)
    
    for epoch in range(EPOCH):
        net.train()
        progress = ProgressMeter(len(dl), prefix="Epoch: [{}]".format(epoch))
        
        for i, data in enumerate(dl):
            points = data["pcd"].to(DEVICE)
            depth = data["depthmap"].to(DEVICE)
            loss = net(depth=depth, points=points)
            progress.update(i, {"loss": loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                progress.display(i)
        scheduler.step()

        torch.save({'epoch': epoch, 
                    'model_state_dict': net.state_dict()},
                    os.path.join(SAVE,'checkpoint_{:03d}.pth'.format(epoch)))
    

if __name__ == "__main__":
    main()
