import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from itertools import chain
import os
import sys
import numpy as np
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from votenet import VoteNet
from sunrgbd_detection_dataset import SunrgbdDatasetConfig, SunrgbdDetectionVotesDataset
from scannet.scannet_detection_dataset import ScannetDatasetConfig, ScannetDetectionDataset
from loss_helper import get_loss

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, help="path to check point")
parser.add_argument("--save", type=str, default="000", help="surfix for log file")
parser.add_argument("--warmup", type=int, default=10, help="epochs for warm up")
parser.add_argument("--dataset", type=str, default="sunrgbd", help="dataset")
parser.add_argument("--percent", type=int, default=-1, help="percentage of training data used. default to 100%")
flags = parser.parse_args()
file = os.path.join("log", "log_finetune_{:s}.txt".format(flags.save))

if flags.ckpt is not None:
    backbone_ckpt = torch.load(flags.ckpt)
LOG_FOUT = open(file, 'a')
WARMUP = flags.warmup
MAX_EPOCH = 180
MILESTONE = [m-WARMUP for m in [80, 120, 160]]
DATASET = flags.dataset
PERCENT = flags.percent

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if DATASET == "sunrgbd":
    config = SunrgbdDatasetConfig()
    ds_train = SunrgbdDetectionVotesDataset(split_set="train",use_v1=True, augment=True)
    ds_val = SunrgbdDetectionVotesDataset(split_set="val", use_v1=True, augment=False)
elif DATASET == "scannet":
    config = ScannetDatasetConfig()
    ds_train = ScannetDetectionDataset(split_set="train", num_points=40000, augment=True)
    ds_val = ScannetDetectionDataset(split_set="val", num_points=40000, augment=False)
else:
    raise ValueError("unknow dataset")

if PERCENT > 0:
    ds_train.sample(PERCENT)

net = VoteNet(num_class=config.num_class,
                num_heading_bin=config.num_heading_bin,
                num_size_cluster=config.num_size_cluster,
                mean_size_arr=config.mean_size_arr,
                input_feature_dim=0,
                num_proposal=256)
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':config}

net = net.cuda()
if flags.ckpt is not None:
    net.backbone_net.load_state_dict(backbone_ckpt["model_state_dict"])

optimizer1 = optim.Adam(chain(net.pnet.parameters(), net.vgen.parameters()), lr=0.001)
optimizer2 = optim.Adam(net.parameters(), lr=0.001)

lr_scheduler = lr_scheduler.MultiStepLR(optimizer2, milestones=MILESTONE, gamma=0.1)

dl_train = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
dl_val = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)
criterion = get_loss

# first stage
for epoch_index in range(WARMUP):
    log_string('**** EPOCH %03d ****' % (epoch_index))
    stat_dict = {} # collect statistics
    np.random.seed()
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(dl_train):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda()

        # Forward pass
        optimizer1.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, config)
        loss.backward()
        optimizer1.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key or 'offset' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
    
    if epoch_index == 0 or epoch_index%10 == 9:
        stat_dict = {} # collect statistics
        ap_calculator = APCalculator(ap_iou_thresh=0.25,
            class2type_map=config.class2type)
        net.eval() # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_label in enumerate(dl_val):
            if batch_idx % 10 == 0:
                print('Eval batch: %d'%(batch_idx))
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].cuda()
            
            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            with torch.no_grad():
                end_points = net(inputs)

            # Compute loss
            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, config)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key or 'offset' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        for key in sorted(stat_dict.keys()):
            log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        # Evaluate average precision
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

# second stage
for epoch_index in range(WARMUP, MAX_EPOCH):
    log_string('**** EPOCH %03d ****' % (epoch_index))
    stat_dict = {} # collect statistics
    net.train() # set model to training mode
    np.random.seed()
    for batch_idx, batch_data_label in enumerate(dl_train):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda()

        # Forward pass
        optimizer2.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, config)
        loss.backward()
        optimizer2.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key or 'offset' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
    
    lr_scheduler.step()
    
    if epoch_index == 0 or epoch_index%10 == 9:
        stat_dict = {} # collect statistics
        ap_calculator = APCalculator(ap_iou_thresh=0.25,
            class2type_map=config.class2type)
        net.eval() # set model to eval mode (for bn and dp)
        np.random.seed()
        for batch_idx, batch_data_label in enumerate(dl_val):
            if batch_idx % 10 == 0:
                print('Eval batch: %d'%(batch_idx))
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].cuda()
            
            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            with torch.no_grad():
                end_points = net(inputs)

            # Compute loss
            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, config)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key or 'offset' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Log statistics
        # TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        #    (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
        for key in sorted(stat_dict.keys()):
            log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        # Evaluate average precision
        log_string("####### AP 25 #######")
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

        log_string("####### AP 50 #######")
        ap_calculator.ap_iou_thresh = 0.5
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))
