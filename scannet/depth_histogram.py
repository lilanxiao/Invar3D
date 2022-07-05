# This experiment shows that each depth map has unique depth histogram
# augmentations like crop and rotatiing cannot significantly change the histogram
# in pretraining, the network could overfit the depth histogram rather than informative 
# features.
# distort depth should address this issue.

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from copy import deepcopy
from scannet_pretrain import ScanNetContrastBase
from rgbd_image import RGBDImage
from data_utils import create_pcd, create_frame, random_distort_depth


def revert_depth(arr : np.ndarray, max_depth = 8.):
    zero_mask = (arr > 1e-2).astype(arr.dtype)
    rarr = max_depth - arr
    return zero_mask * rarr


def main():
    np.random.seed(10)
    
    ds = ScanNetContrastBase("train")
    depth, intrin = ds[np.random.randint(0, len(ds))]
    img = RGBDImage(depth=depth, intrinsic=intrin, rgb=None)
    img2 = deepcopy(img)
    img.rotate(np.random.rand()*10 - 5)
    img.random_box_erase(0.2, 0.4)
    img.random_crop(0.7, 0.9)
    img.letter_box((352, 352))
    
    img2.rotate(np.random.rand()*10 - 5)
    img2.random_box_erase(0.2, 0.4)
    img2.random_crop(0.7, 0.9)
    img2.letter_box((352, 352))
    
    np.random.seed()
    img.depth = random_distort_depth(img.depth, 0.3)
    
    
    depth1 = img.depth
    depth2 = img2.depth
    
    # depth2 = revert_depth(depth2)
    
    plt.figure()
    plt.imshow(depth1)
    plt.show()
    
    
    plt.figure()
    plt.hist(np.reshape(depth1, (-1, )), bins=50, alpha = 0.5, range=(0, 8))
    plt.hist(np.reshape(depth2, (-1, )), bins=50, alpha = 0.5, range=(0, 8))
    plt.show()
    
    # img.depth = revert_depth(img.depth)
    
    pts = img.to_pointcloud()
    pts = create_pcd(pts)
    frame = create_frame()
    o3d.visualization.draw_geometries([pts, frame])


if __name__ == "__main__":
    main()
    