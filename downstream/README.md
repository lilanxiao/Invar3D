# VoteNet finetuning

This document shows how to finetune a VoteNet with the pretrained weights. 

- Use `extract_dp_ddp_model.py` to extract pretrained weights of PointNet++ from a DDP model. This script is for models trained with `train_ddp_moco_ddp.py`. For other pre-training methods, the required modification is straightforward and is thus not provided here. 
- Clone the VoteNet [official repo]("https://github.com/facebookresearch/votenet") and follow it's instruction to install dependencies and prepare finetuning data. 
- Put `votenet_finetune.py` to the root of VoteNet repo.
- Add following method to `votenet/sunrgbd/sunrgbd_detection_dataset.py` and `votenet/scannet/scannet_detection_dataset.py`. This is used to sample the training data for the data efficiency experiment. 

        def sample(self, percentage:int = 100):
            # fix seed
            np.random.seed(int(percentage))
            choice = np.random.permutation(len(self))[:int(percentage/100*len(self))]
            self.scan_names = [self.scan_names[i] for i in choice]
            print("sampled {:d} scans from {:s} set".format(len(self), self.data_path))
            # reset it back
            np.random.seed()

- You can use following flags in finetuning: 
```
    --ckpt: checkpoint path of the pretrianed pointnet++ model
    --save: save path for log files, ect.
    --warmup: number of epochs to warmup. during warmup, the backbone is not trained. 
    --dataset: choose sunrgbd or scannet.
    --percent: percentage of data to use for training. set it to -1 to use all data. 
```