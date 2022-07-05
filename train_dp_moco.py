# ------------------------------------------------
import cv2      # first import cv2, then torch
cv2.setNumThreads(0)
# https://github.com/pytorch/pytorch/issues/1838
# no need to use OpenCV with multithreading since 
# Pytorch alread uses multiprocessing 
# ------------------------------------------------
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from itertools import chain

from loss_moco import NCELossMocoV2, UNetHead, PointNet2Head, momentum_update
from scannet.scannet_pretrain import ScannetDepthPointGlobalDataset
from model.cnn2d.conv2d import UNet
from model.pointnet2.backbone_module import Pointnet2Backbone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8, help="batch size [default:8]")
parser.add_argument("--epoch", type=int, default=30, help="epochs to train [default:30]")
parser.add_argument("--temp", type=float, default=0.07, help="temperature of InfoCNE loss for local features [default:0.07]")
parser.add_argument("--save", type=str, default="log/depth_point_global", help="save path [default:log/depth_point_global]")
flags = parser.parse_args()

BATCH_SIZE = flags.batch_size
EPOCH = flags.epoch
TEMPERATURE = flags.temp
SAVE_PATH = os.path.join(BASE_DIR, flags.save)
NUM_NEGATIVES_MOCO = 4096*9
DIM_OUT = 128
DIM_INTER = 512
LR = 0.1
SAVE_INTERVAL = 1

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    net2d = UNetHead(UNet(), 256, DIM_INTER, DIM_OUT).cuda()
    net2d_m = UNetHead(UNet(), 256, DIM_INTER, DIM_OUT).cuda()
    net3d = PointNet2Head(Pointnet2Backbone(), 256, DIM_INTER, DIM_OUT).cuda()
    net3d_m = PointNet2Head(Pointnet2Backbone(), 256, DIM_INTER, DIM_OUT).cuda()
    loss_func1 = NCELossMocoV2(NUM_NEGATIVES_MOCO, DIM_OUT, TEMPERATURE, normalize_embedding=[True, True]).cuda()
    loss_func2 = NCELossMocoV2(NUM_NEGATIVES_MOCO, DIM_OUT, TEMPERATURE, normalize_embedding=[True, True]).cuda()
    ds = ScannetDepthPointGlobalDataset("train")
    dataloader = DataLoader(ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=6, 
                            pin_memory=True,
                            worker_init_fn=my_worker_init_fn,
                            drop_last=True)
    optimizer = optim.SGD(chain(net2d.parameters(), net3d.parameters()), 
                          lr=LR, 
                          momentum=0.9,
                          weight_decay=1e-4
                          )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, LR/1000)

    for idx_epoch in range(1, EPOCH+1):
        net2d.train()
        net2d_m.train()
        net3d.train()
        net3d_m.train()
        losses = AverageMeter("loss")
        losses23 = AverageMeter("loss23")
        losses32 = AverageMeter("loss32")        
        for idx_iter, data in enumerate(dataloader, 1):
            for key in data:
                data[key] = data[key].cuda()
            optimizer.zero_grad()
            # keys
            k2d = net2d(data["depthmap1"], data["feature_map_mask1"])
            k3d = net3d(data["pcd1"])
            # queries and momentum model
            with torch.no_grad():
                q2d = net2d_m(data["depthmap2"], data["feature_map_mask2"])
                q3d = net3d_m(data["pcd2"])
            # 2d model generates keys, 3d model generates queries
            loss23 = loss_func1(k2d, q3d)
            # 2d model generates queries, 3d model generates keys
            loss32 = loss_func2(k3d, q2d)
            loss = (loss23 + loss32)/2
            loss.backward()
            # update models
            optimizer.step()
            # update momentum models
            momentum_update(net2d, net2d_m)
            momentum_update(net3d, net3d_m)
            # statistics
            losses.update(loss.item())
            losses23.update(loss23.item())
            losses32.update(loss32.item())
            
            if idx_iter % 20 == 0:
                print("[Ep.{:d}: {:d}/{:d}] {:s}; {:s}; {:s}".format(
                    idx_epoch,
                    idx_iter,
                    len(ds)//BATCH_SIZE,
                    str(losses),
                    str(losses23),
                    str(losses32)
                ))
        
        scheduler.step()
        
        if idx_epoch % SAVE_INTERVAL == 0:
            torch.save({'epoch': idx_epoch, 
                        'model_state_dict': net2d.encoder.state_dict()},
                        os.path.join(SAVE_PATH,'ep{:03d}_loss{:.5f}_depth.pth'.format(idx_epoch, losses.avg)))
            torch.save({'epoch': idx_epoch, 
                        'model_state_dict': net3d.encoder.state_dict()},
                        os.path.join(SAVE_PATH,'ep{:03d}_loss{:.5f}_pcd.pth'.format(idx_epoch, losses.avg)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.3f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
    