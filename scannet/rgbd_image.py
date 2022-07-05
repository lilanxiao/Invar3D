import cv2 
import numpy as np
from PIL import Image
from numpy import ndarray
from data_utils import (
    box_erase,depth2points, remove_zeros, rotate_img, 
    downsample_img_calib, letter_box, random_crop,
    flip_image, create_pcd)


class RGBDImage(object):
    def __init__(self, rgb:ndarray, depth:ndarray, intrinsic:ndarray) -> None:
        super().__init__()
        assert rgb is not None or depth is not None, "one of rgb and depth must be given"
        self.rgb = rgb
        self.depth = depth
        self.intrinsic = intrinsic
        if self.has_rgb:
            self.color_dim = rgb.shape[2]
        
        if self.has_depth and self.has_rgb:
            assert depth.shape[0] == rgb.shape[0]
            assert depth.shape[1] == rgb.shape[1]
        if self.has_rgb:
            assert (self.rgb.dtype == np.float32 or 
                    self.rgb.dtype == np.float64), "please use float type rgb"
    
    @property    
    def has_rgb(self):
        return self.rgb is not None
    
    @property
    def has_depth(self):
        return self.depth is not None

    def cat(self):
        return np.concatenate([self.rgb, np.expand_dims(self.depth, 2)], 2)      
    
    def split(self, rgbd):
        rgb = rgbd[:, :, :self.color_dim]
        depth = rgbd[:, :, self.color_dim]
        return rgb, depth
    
    def rotate(self, angle:float, center=None) -> None:
        """rotate image. 

        Args:
            angle (float): angle in degree. 
            center (optional): rotation center. If None, rotate around principle point.
        """
        if center is None:
            # rotate around principle point
            c = [self.intrinsic[0, 2], self.intrinsic[0, 1]]
        if self.has_rgb:
            self.rgb = rotate_img(self.rgb, angle, center=c, flags=cv2.INTER_CUBIC)
        if self.has_depth:
            self.depth = rotate_img(self.depth, angle, center=c, flags=cv2.INTER_NEAREST)
    
    def flip_h(self) -> None:
        """horizontal flip, update intrinsic
        """
        temp = np.copy(self.intrinsic)
        if self.has_depth:
            self.depth, self.intrinsic = flip_image(self.depth, temp)
        if self.has_rgb:
            self.rgb, self.intrinsic = flip_image(self.rgb, temp)
    
    def letter_box(self, size):
        """resize to targe size while keeping aspect ratio by zero padding.
        """
        temp = np.copy(self.intrinsic)
        if self.has_depth:
            self.depth, self.intrinsic = letter_box(self.depth, temp, size, flags=cv2.INTER_NEAREST)
        if self.has_rgb:
            self.rgb, self.intrinsic = letter_box(self.rgb, temp, size, flags=cv2.INTER_CUBIC)
    
    def random_crop(self, min_ratio:float, max_ratio:float):
        """random crop image
        """
        if min_ratio > 1 or max_ratio > 1:
            raise ValueError("crop ratio cannot be greater than 1")
        temp = np.copy(self.intrinsic)
        if self.has_depth and self.has_rgb:
            rgbd = self.cat()
            rgbd, self.intrinsic = random_crop(rgbd, temp, min_ratio, max_ratio)
            self.rgb, self.depth = self.split(rgbd)
        elif self.has_depth:
            self.depth, self.intrinsic = random_crop(self.depth, temp, min_ratio, max_ratio)
        elif self.has_rgb:
            self.rgb, self.intrinsic = random_crop(self.rgb, temp, min_ratio, max_ratio)
    
    def downsampled_depth(self, factor:int):
        if factor < 1:
            raise ValueError("downsample factor muss be integral greater than 1")
        temp = np.copy(self.intrinsic)
        if self.has_depth:
            return downsample_img_calib(self.depth, temp, factor)
        else:
            raise ValueError("this rgbd image object has no depth map")
    
    def random_box_erase(self, low, high, same=True):
        if self.has_depth and self.has_rgb and same:
            rgbd = self.cat()
            rgbd = box_erase(rgbd, low, high)
            self.rgb, self.depth = self.split(rgbd)
        elif self.has_depth:
            self.depth = box_erase(self.depth, low, high)
        elif self.has_rgb:
            self.rgb = box_erase(self.rgb, low, high)
    
    def to_pointcloud(self, return_zero_mask = False):
        """project rgbd image to point cloud

        Returns:
            points: Nx6 if rgb exists, else Nx3
        """
        if not self.has_depth:
            raise ValueError("cannot project rgbd to depth as depth not exist")
        pts = depth2points(self.depth, self.intrinsic)
        pts, mask = remove_zeros(pts, return_mask=True)
        zero_mask = np.nonzero(mask)[0]
        if self.has_rgb:
            rgb_flat = np.reshape(self.rgb, (-1, 3))
            rgb_flat = rgb_flat[zero_mask, :]
            pts = np.concatenate([pts, rgb_flat], 1)
        if return_zero_mask:
            return pts, zero_mask
        else:
            return pts
    
    def transform_rgb(self, transform):
        """transform rgb image"""
        dt = self.rgb.dtype
        rgb = (self.rgb * 255).astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb = transform(rgb)
        rgb = np.asarray(rgb)
        self.rgb = (rgb / 255).astype(dt)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import open3d as o3d
    from scannet_pretrain import ScanNetContrastBase
    ds = ScanNetContrastBase("train")
    depth, intrinsic = ds[0]
    rgbd = RGBDImage(rgb=np.random.rand(depth.shape[0], depth.shape[1], 3), depth=depth, intrinsic=intrinsic)
    
    def show(img:RGBDImage):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img.rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(img.depth)
        plt.show()
    
    show(rgbd)

    rgbd.rotate(30)
    show(rgbd)    
    
    rgbd.random_crop(0.4, 0.8)
    show(rgbd)
    
    rgbd.random_box_erase(0.2, 0.4, True)
    show(rgbd)
    
    rgbd.flip_h()
    show(rgbd)
    
    pcd = rgbd.to_pointcloud()
    pcdo3d = create_pcd(pcd[:, :3], pcd[:, 3:])
    o3d.visualization.draw_geometries([pcdo3d])
        
        