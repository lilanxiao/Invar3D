# A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision

This repository is for our paper "A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision", Lanxiao Li and Michael Heizmann, ECCV 2022. 

## Requirements

### Hardware

The training scripts with `_ddp` expect distributed training with multiple GPUs. But some samples for single GPU training is also provided. 

### Software

The repo is tested under Ubuntu 18.04 and 20.04. CUDA-toolkit (tested with 10.2 and 11.1) and GCC is needed to compile some extensions. Also, following python packages are required:

    pytorch     # tested with 1.8. other versions should work as well
    torchvision
    open3d
    matplotlib
    scipy
    pybind11
    opencv-python
    pillow
    MinkowskiEngine=0.5.4

To install MinkowskiEngine, please follow the [official repo](https://github.com/NVIDIA/MinkowskiEngine). 


## Preparation

### Extensions

To compile C++-extensions, go to `cpp_ext/fps` and `cpp/knn` and run 

    bash build.sh

in **each** folder. 

To compile CUDA-extension (PointNet++), go to `model/pointnet2` and run 

    python setup.py install


### Data

To prepare the pre-training data: 

- First you need to download and prepare the ScanNet raw data. Please follow `README.md` in `prepare_data`.
- Update the data path in `scannet/config`. The data folder should have the following structure:

        data_folder
        |
        |__ scene0000_00
        |   |
        |   |__ _info.txt               # meta data of the scene
        |   |__ frame-000000.color.jpg  # color image, resized
        |   |__ frame-000000.png        # depth map
        |   |__ frame-000000.pose.txt   # camera pose. unused. 
        |   |__ frame-000001.color.jpg
        |   |__ frame-000001.png
        |   |__ frame-000001.pose.txt
        |   |   ... ...
        |
        |__ scene0000_01
        |__ scene0000_02
        |   ... ... 


- You don't need to sample the data. The sampled `frame-IDs` (with the factor 25) are already provided in `scannet/sampled_train_25.txt`. The code in `scannet/sampler.py` is used for sampling.
- **[Optional]** If you want to save the sampled data to another place, use `scannet/save_sampled`. Remember to update `scannet/config` if you want to read data from this new place. 
- **[Notice]** Some extracted depth maps might only contains NaN or O value and are thus invalid. We've found all invalid `frame-Ids` and saved them in `scannet/config.py`. These frames are not used for training.   

## Usage

### Pretraining

To pretrain a PointNet++ and a depth map based CNN (DPCo), use

    python train_dp_moco_ddp.py \
    --lr 0.03 \
    --save log/DPCo \
    --batch-size 64 \
    --cos \
    --local \
    --moco \
    --worker 8 \
    --epochs 120 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0

Our training is done on a single node with 2 NVIDIA Tesla V100 GPUs. You might have to update some parameters (e.g. workers, batch-size, world-size) according to you own hardware. Also, the code for single GPU without DDP is provided in `train_ddp_moco.py`. But we only use this version for debugging purpose. 

Similarly, to pretrain a sparse 3D CNN and depth map based CNN (DVCo with color), use

    export OMP_NUM_THREADS=12 # make MinkowskiEngine happy
    python train_dv_ddp.py \
    --lr 0.03 \
    --save log/DVCo \
    --batch-size 64 \
    --cos \
    --moco \
    --local \
    --worker 8 \
    --epochs 120 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0


### Finetuning

For finetuning on 3D object detection task, please follow `README.md` in `downstream`. 

## Note

We are still working on cleaning our internal code base and testing with this public repo. There would be updates in the future. 

## Known Issues

- We encountered OOM problems with MinkowskiEngine. The CPU RAM usage increased constantly with some of our code. Current workaround: Manually pause and resume the training to release the RAM. 
- The training might stop or become very slow sometimes, because the Dataset class tries to find more unique local correspondences and get stuck. In this case, try to decrease the ratio of unique matched points or increase the matching threshold `match_thresh`, as commented in `scannet/scannet_pretrain.py`. 

## Citation

If you find this repo helpful, please consider cite our work 

    @inproceedings{li2022invar3d,
        author = {Li, Lanxiao and Heizmann, Michael},
        title = {A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision},
        booktitle = {ECCV},
        year = {2022}
    }

## Acknowledgement

This repo has modified some code from following repos. We thank the authors for their amazing code bases. Please consider star/cite their works as well. 

- ScanNet: https://github.com/ScanNet/ScanNet
- MinkowskiEngine: https://github.com/NVIDIA/MinkowskiEngine
- MoCo: https://github.com/facebookresearch/moco
- VoteNet: https://github.com/facebookresearch/votenet
- DepthContrast: https://github.com/facebookresearch/DepthContrast
