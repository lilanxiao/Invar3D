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
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import MinkowskiEngine as ME

from scannet.scannet_pretrain import ScanNetDepthVoxelDataset
from utils.sparse_helper import to_device
from train_dv import DepthVoxelContrast
from meters import AverageMeter, ProgressMeter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default=0.07, help="temperature of InfoCNE loss [default:0.07]")
parser.add_argument("--save", type=str, default="log/DVCo", help="save path")
parser.add_argument("--local", action="store_true", help="use PointINC loss (PointContrast)")
parser.add_argument("--moco", action="store_true", help="use MoCo-style loss for global features")
parser.add_argument("--within-format", action="store_true", help="MOCO loss within format")
parser.add_argument("--asymm", action="store_true", help="use asynmmetric loss")
parser.add_argument("--warmup", type=int, default=-1, help="use low weighting for local loss in early stage")
parser.add_argument("--voxel-size", type=float, default=0.025, help="voxel size of sparse CNN")

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
    model = DepthVoxelContrast(
        dim_out=128, 
        dim_inter=512,
        num_neg=4096*8,
        temperature=0.07,
        local_loss=args.local,
        global_loss=args.moco,
        within_format=args.within_format,
        ddp=True,
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
    train_dataset = ScanNetDepthVoxelDataset("train", num_pairs=2, voxel_size=args.voxel_size)
    val_dataset = ScanNetDepthVoxelDataset("val", num_pairs=2, voxel_size=args.voxel_size)
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True, 
        collate_fn=my_collate_fn)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=True, 
        collate_fn=my_collate_fn)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)
        
        # NOTE: hack to release memory occupied by MinkowskiEngine
        # NOT real validation. Only used to avoid OOM
        fake_val(val_loader, model, args)

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
            to_device(data, args.gpu)
        data["sin1"] = ME.SparseTensor(
            data["feats1"], data["coords1"]
        )
        data["sin2"] = ME.SparseTensor(
            data["feats2"], data["coords2"]
        )
        
        loss, metric  = model(data, epoch+1)
        progress.update(i, metric)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update momentum model
        model.module.update_momentum_models()

        if i % 20 == 0:
            torch.cuda.empty_cache()

        if i % args.print_freq == 0:
            progress.display(i)

     
def fake_val(val_loader, model, args):
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            to_device(data, args.gpu)
            data["sin1"] = ME.SparseTensor(
                data["feats1"], data["coords1"]
            )
            data["sin2"] = ME.SparseTensor(
                data["feats2"], data["coords2"]
            )
            out = model.module.net3d(data["sin1"], data["ind_vox1"])
            # do nothing
            if i > 20:
                break


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
            

def my_collate_fn(data_list):
    # nested function doesn't work with multiprocessing
    sparse_key = ["coords", "feats"]
    pairs = 2
    
    keys = data_list[0].keys()
    ret = {key:[] for key in keys}
    for data in data_list:
        for key in keys:
            ret[key].append(torch.from_numpy(data[key]))
    # stack dense tensor
    for key in keys:
        if sparse_key[0] not in key and sparse_key[1] not in key:
            ret[key] = torch.stack(ret[key], axis=0)
    # collate sparse tensor
    if pairs == 1:
        coord, feat = ME.utils.sparse_collate(ret[sparse_key[0]], ret[sparse_key[1]])
        ret[sparse_key[0]] = coord
        ret[sparse_key[1]] = feat
    else:
        for i in range(1, pairs+1):
            coord, feat = ME.utils.sparse_collate(ret[sparse_key[0]+str(i)], ret[sparse_key[1]+str(i)])
            ret[sparse_key[0]+str(i)] = coord
            ret[sparse_key[1]+str(i)] = feat
        return ret


if __name__ == "__main__":
    main()
    