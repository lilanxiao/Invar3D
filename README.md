# A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision

Offcial repository for "A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision", ECCV 2022. 

## Notice

This repository is not fully tested yet. Still working on clearning the code. 

## Requirements

### Hardware

The training scripts with `_ddp` expect distributed training with multiple GPU. But some samples for single GPU training is also provided. 

### Software

The code is tested under Ubuntu 18.04 and 20.04. CUDA-toolkit (tested with 10.2 and 11.1) and GCC is needed for compiling some extensions. Also, following python packges are required:

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

under in **each** folder. 

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
- **[Optional]** if you want to save the sampled data to another place, use `scannet/save_sampled`. Remeber to update `scannet/config` if you want to read data from this new place. 



