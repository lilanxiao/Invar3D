# ------------------------------------------------
import cv2      # first import cv2, then torch
cv2.setNumThreads(0)
# https://github.com/pytorch/pytorch/issues/1838
# ------------------------------------------------
import os
import math
import argparse
import builtins
import shutil
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from loss_moco import NCELossMocoV2, UNetHead, PointNet2Head, momentum_update, batch_shuffle_ddp, batch_unshuffle_ddp
from loss import point_info_nce_loss, _gather
from scannet.scannet_pretrain import ScannetDepthPointGlobalDataset, ScannetDepthPointDataset
from model.cnn2d.conv2d import UNet
from model.pointnet2.backbone_module import Pointnet2Backbone
from meters import AverageMeter, ProgressMeter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default=0.07, help="temperature of InfoCNE loss [default:0.07]")
parser.add_argument("--save", type=str, default="log/DPCo", help="save path")
parser.add_argument("--local", action="store_true", help="use PointINC loss (PointContrast)")
parser.add_argument("--moco", action="store_true", help="use MoCo-style loss for global features")
parser.add_argument("--asymm", action="store_true", help="use asynmmetric loss")
parser.add_argument("--dense-proj", action="store_true", help="use dense projection head")
parser.add_argument("--warmup", type=int, default=-1, help="use low weighting for local loss in early stage")
parser.add_argument("--within-format", action="store_true", help="MOCO loss within format")

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

DIM_OUT = 128
DIM_INTER = 512
NUM_NEG = 4096*8

def main():
    args = parser.parse_args()    
    SAVE_PATH = os.path.join(BASE_DIR, args.save)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    args.save = SAVE_PATH

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = DepthPointMOCO(dim_out=DIM_OUT,
                           dim_inter=DIM_INTER, 
                           num_neg=NUM_NEG, 
                           temperature=args.temp, 
                           local_loss=args.local,
                           global_loss=args.moco,
                           asymm=args.asymm,
                           dense_head=args.dense_proj,
                           within_format=args.within_format,
                           warmup=args.warmup
                           )
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.asymm)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=args.asymm)        
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = optim.SGD(model.parameters(), args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # train_dataset = ScannetDepthPointGlobalDataset("train", augment=True)
    train_dataset = ScannetDepthPointDataset("train", augment=True, num_pairs=2)
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{:s}/checkpoint_{:04d}.pth.tar'.format(args.save, epoch))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    

def train(train_loader, model, optimizer, epoch, args):
    progress = ProgressMeter(
        len(train_loader),
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for i, data in enumerate(train_loader):
        if args.gpu is not None:
            for key in data:
                data[key] = data[key].cuda(args.gpu, non_blocking=True)
        
        loss, metric  = model(data, epoch+1)
        progress.update(i, metric)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update momentum model
        model.module.update_momentum_models()

        if i % args.print_freq == 0:
            progress.display(i)
     

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


from typing import Dict
class DepthPointMOCO(nn.Module):
    def __init__(self, 
                dim_out, 
                dim_inter, 
                num_neg, 
                temperature, 
                momentum=0.999, 
                local_loss=False,
                global_loss=True, 
                asymm=False,
                dense_head=False,
                within_format=False,
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
        self.asymm = asymm
        self.within_format = within_format
        if self.global_loss and self.local_loss:
            self.warmup = warmup
        else:
            self.warmup = -1
        # initialize models
        self.net2d = UNetHead(UNet(), 256, dim_inter, dim_out, dense_head)
        self.net2d_m = UNetHead(UNet(), 256, dim_inter, dim_out, dense_head)
        self._copy(self.net2d, self.net2d_m)
        self.net3d = PointNet2Head(Pointnet2Backbone(), 256, dim_inter, dim_out, dense_head)
        self.net3d_m = PointNet2Head(Pointnet2Backbone(), 256, dim_inter, dim_out, dense_head)
        self._copy(self.net3d, self.net3d_m)
        # initialize queues
        if not self.asymm:
            self.loss_func1 = NCELossMocoV2(num_neg, dim_out, temperature, normalize_embedding=[True, True])
        self.loss_func2 = NCELossMocoV2(num_neg, dim_out, temperature, normalize_embedding=[True, True])
        if within_format:
            self.loss_func3 = NCELossMocoV2(num_neg, dim_out, temperature, normalize_embedding=[True, True])
            self.loss_func4 = NCELossMocoV2(num_neg, dim_out, temperature, normalize_embedding=[True, True])
        
    def _copy(self, model:nn.Module, model_m:nn.Module):
        for p, pm in zip(model.parameters(), model_m.parameters()):
            pm.data.copy_(p.data)
            pm.requires_grad = False
    
    def update_momentum_models(self):
        momentum_update(self.net2d, self.net2d_m, self.momentum)
        momentum_update(self.net3d, self.net3d_m, self.momentum)
        
    def symmetric_contrast(self, data:Dict[str, torch.Tensor], epoch=None):
        q2d, _, f2d = self.net2d(data["depthmap1"], data["feature_map_mask1"], True)
        q3d, f3d = self.net3d(data["pcd1"], data["pcd_fps_ind1"], True)
        
        # PointContras-style loss: correspondence learning
        f2d = _gather(f2d, data["ind_res1"])
        f3d = _gather(f3d, data["ind_fps1"])
        if self.local_loss:
            f2d = torch.nn.functional.normalize(f2d, dim=1)
            f3d = torch.nn.functional.normalize(f3d, dim=1)
            loss_pc = point_info_nce_loss(f2d, f3d, self.t)["loss"]
        else:
            loss_pc = loss_pc = torch.tensor(0.).to(q2d.device)
        
        # queries and momentum model
        with torch.no_grad():
            dm = data["depthmap2"]
            fmm = data["feature_map_mask2"]
            pcd = data["pcd2"]
            # BN shuffle for keys
            dm, idx_unshuffle, idx_shuffle = batch_shuffle_ddp(dm, return_idx_shuffle=True)
            fmm, _ = batch_shuffle_ddp(fmm, idx=idx_shuffle)
            pcd, idx_unshuffle2 = batch_shuffle_ddp(pcd)
            # forward
            k2d = self.net2d_m(dm, fmm)
            k3d = self.net3d_m(pcd)
            # unshuffle
            k2d = batch_unshuffle_ddp(k2d, idx_unshuffle)
            k3d = batch_unshuffle_ddp(k3d, idx_unshuffle2)
            
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
        
        if not self.global_loss:
            loss_g = torch.tensor(0.).to(q2d.device)

        if epoch is not None and self.warmup > 0:
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
        return loss, metric 
    
    def asymmetric_contrast(self, data:Dict[str, torch.Tensor], epoch=None):
        # NOTE: have to use local and global loss, when asymmetric is activated
        _, _, f2d = self.net2d(data["depthmap1"], data["feature_map_mask1"], True)
        q3d, _ = self.net3d(data["pcd1"], data["pcd_fps_ind1"], True)
        
        with torch.no_grad():
            _, f3d = self.net3d_m(data["pcd1"], data["pcd_fps_ind1"], True)
        
        # PointContras-style loss: correspondence learning
        # this loss only update 2d net
        f2d = _gather(f2d, data["ind_res1"])
        f3d = _gather(f3d, data["ind_fps1"])
        f2d = torch.nn.functional.normalize(f2d, dim=1)
        f3d = torch.nn.functional.normalize(f3d, dim=1)
        loss_pc = point_info_nce_loss(f2d, f3d, self.t)["loss"]
                         
        # queries and momentum model
        # this loss only update 3d net
        with torch.no_grad():
            dm = data["depthmap2"]
            fmm = data["feature_map_mask2"]
            # BN shuffle for keys
            dm, idx_unshuffle, idx_shuffle = batch_shuffle_ddp(dm, return_idx_shuffle=True)
            fmm, _ = batch_shuffle_ddp(fmm, idx=idx_shuffle)
            # forward
            k2d = self.net2d_m(dm, fmm)
            # unshuffle
            k2d = batch_unshuffle_ddp(k2d, idx_unshuffle)

        # 3d model generates queries, 2d model generates keys
        loss32 = self.loss_func2(q3d, k2d)
        
        loss_g = loss32
        if epoch is not None and self.warmup > 0:
            wl = 0.5 * min(epoch/self.warmup, 1)
            wg = 1 - wl
            loss = loss_g * wg + loss_pc * wl
        else:           
            loss = ( loss_g + loss_pc ) *0.5
        
        metric = {"loss_g":loss_g.item(),
                  "loss": loss.item(),
                  "loss_pc": loss_pc.item()}
        return loss, metric          
    
    def forward(self, data:Dict[str, torch.Tensor], epoch=None):
        if self.asymm:
            return self.asymmetric_contrast(data, epoch=epoch)
        else:
            return self.symmetric_contrast(data, epoch=epoch)
            

if __name__ == "__main__":
    main()
    