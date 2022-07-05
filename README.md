# A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision

This repository is for our paper "A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision", ECCV 2022. 

## Requirements

### Hardware

The training scripts with `_ddp` expect distributed training with multiple GPUs. But some samples for single GPU training is also provided. 

### Software

The code is tested under Ubuntu 18.04 and 20.04. CUDA-toolkit (tested with 10.2 and 11.1) and GCC is needed for compiling some extensions. Also, following python packages are required:

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


- You don't need to sample the data. The sampled `frmae-IDs` (with factor 25) are already provided in `scannet/sampled_train_25.txt`. The code in `scannet/sampler.py` is used for sampling.
- **[Optional]** if you want to save the sampled data to another place, use `scannet/save_sampled`. Remember to update `scannet/config` if you want to read data from this new place. 


## Usage

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

The training is done on a single node with 2 NVIDIA Tesla V100 GPU. You might have to update some parameters (e.g. workers, batch-size, work-size) according to you own hardware. Also, the code for single GPU without DDP is in `train_ddp_moco.py`. But this version is only for debugging purpose. 

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

## Note

We are still working on cleaning our internal code base and testing with this public repo. There would be updates in the future. 

## Known Issues

- We encountered OOM problems with MinkowskiEngine. The CPU RAM usage increased constantly with some of our code. Current workaround: Manually pause and resume the training to release the RAM. 
- The training might stop or become very slow sometimes, because the Dataset class tries to find more unique local correspondences and get stuck. In this case, try to decrease the ratio of unique matched points, as commented in `scannet/scannet_pretrain.py`. 

## Citation

If you find this repo helpful, please consider cite our work 

    @inproceedings{li2022invar3d,
        author = {Li, Lanxiao and Heizmann, Michael},
        title = {A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision},
        booktitle = {ECCV},
        year = {2022}
    }

