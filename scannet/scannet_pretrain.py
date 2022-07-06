import os
import cv2
import numpy as np
import sys
import open3d as o3d
try:
    import MinkowskiEngine as ME
except:
    print("Warning! MinkowsiEngine not found")
from copy import deepcopy
from typing import Dict, List, Tuple
from numpy import ndarray
from torch.utils.data import Dataset
from torchvision import transforms as tvt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, "../cpp_ext/fps"))
sys.path.append(os.path.join(BASE_DIR, "../cpp_ext/knn"))
from data_utils import (
    _png2depth, jitter_depth, jitter_pcd, box_erase,
    read_scannet_instri, depth2points, remove_zeros, 
    rotate_img, downsample_img_calib, letter_box, random_crop,
    flip_image, rotz, rotx, roty, create_offset, create_pcd, 
    flip_depth, random_distort_depth)
from cpp_ext.knn.nanoflann import knn_flann
from cpp_ext.fps.fps_np import farthest_point_sample
from model.cnn2d.cam_conv import CAMFeatureas
from rgbd_image import RGBDImage

from config import TARGET, BLACK_LIST
SCANS = sorted(os.listdir(TARGET))
TRAIN_FRAMES = os.path.join(BASE_DIR, "sampled_train_25.txt")
VAL_FRAMES = os.path.join(BASE_DIR, "sampled_val_25.txt")
MAX_DEPTH = 8.


# ------------------------------------------------------------------------------------
# ----------------------------------- BASIC CLASSES ---------------------------------- 
# ------------------------------------------------------------------------------------

class ScanNetContrastBase(object):
    """base class. provide depth map and intrinsic matrix"""
    def __init__(self, split : str, use_rbg = False, frames=None, true_depth=True) -> None:
        super().__init__()
        self.split = split
        self.use_rgb = use_rbg
        self.true_depth = true_depth
        if split == "train":
            file = TRAIN_FRAMES
        elif split == "val":
            file = VAL_FRAMES
        else:
            ValueError("unknow split !")
        if frames is None:
            with open(file, "r") as f:
                self.frames = f.read().splitlines()
        else:
            # allow pass frames as arg
            self.frames = frames
        # some frames are invalid. remove them
        self.frames = [f for f in self.frames if f not in BLACK_LIST]
        self._get_all_intrinsic()
        
    def __len__(self) -> int:
        return len(self.frames)

    def _get_all_intrinsic(self):
        self.intris = {}
        for key in SCANS:
            self.intris[key] = read_scannet_instri(os.path.join(TARGET, key, "_info.txt"))

    def _get_one_frame(self, scan, f):
        depth = _png2depth(os.path.join(TARGET, scan, f + ".png"), self.true_depth)
        if self.use_rgb:
            bgr = cv2.imread(os.path.join(TARGET, scan, f + ".color.jpg"))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # deep copy because it might be changed inplace
        intri =  deepcopy(self.intris[scan])
        if self.use_rgb:
            return depth, rgb, intri
        else:
            return depth, intri
    
    def __getitem__(self, index) -> Tuple[ndarray, ndarray]:
        path = self.frames[index]
        scan, f = path.split("/")
        return self._get_one_frame(scan, f)                


class DepthMapCrop(object):
    """create two cropes of depth map"""
    def __init__(self, diff_crop:bool, low:float=0.7, high:float=0.9, ) -> None:
        super().__init__()
        self.low = low
        self.high = high
        self.diff_crop = diff_crop
    
    def __call__(self, img:ndarray, intrinsic:ndarray) :
        img1, intrinsic1 = random_crop(img, intrinsic, self.low, self.high)
        if self.diff_crop:
            img2, intrinsic2 = random_crop(img, intrinsic, self.low, self.high)
        else:
            img2 = np.copy(img1)
            intrinsic2 = np.copy(intrinsic1)
        return img1, img2, intrinsic1, intrinsic2


class PointCloudAugment(object):
    """augument point cloud with flip, rotate and scale"""
    def __init__(self, delta_scale:float=0.3, delta_angles:List[float]=[np.pi/6, np.pi/6, np.pi/3]) -> None:
        super().__init__()
        assert len(delta_angles) == 3, "for x, y, z axis. need 3 angles."
        self.delta_scale = delta_scale
        self.delta_angles = delta_angles
    
    def __call__(self, input) -> ndarray:
        """support an ndarray or a list of ndarray"""
        if isinstance(input, ndarray):
            # augment a single piont cloud
            pcd = input
            if np.random.random() > 0.5:
                # flipping along the YZ plane
                pcd[:,0] = -1 * pcd[:,0]
            if np.random.random() > 0.5:
                # flipping along the XZ plane
                pcd[:,1] = -1 * pcd[:,1]
            # random rotate around each axis
            for func, angle in zip([rotx, roty, rotz], self.delta_angles):
                rot_angle = np.random.rand()*angle*2 - angle       # -angle ~ +angle
                rot_mat = func(rot_angle)
                pcd[:,0:3] = np.dot(pcd[:,0:3], np.transpose(rot_mat))
            # scale
            scale_ratio = np.random.random()*self.delta_scale * 2 + 1 - self.delta_scale        
            pcd *= scale_ratio
            return pcd
        elif isinstance(input, list):
            # apply same augmentation to a list of point clouds
            n = len(input)
            if np.random.random() > 0.5:
                # flipping along the YZ plane
                for i in range(n):
                    input[i][:,0] = -1 * input[i][:,0]
            if np.random.random() > 0.5:
                # flipping along the XZ plane                
                for i in range(n):
                    input[i][:,1] = -1 * input[i][:,1]
            # random rotate around each axis
            for func, angle in zip([rotx, roty, rotz], self.delta_angles):
                rot_angle = np.random.rand()*angle*2 - angle       # -angle ~ +angle
                rot_mat = func(rot_angle)
                for i in range(n):
                    input[i][:,0:3] = np.dot(input[i][:,0:3], np.transpose(rot_mat))
            # scale
            scale_ratio = np.random.random()*self.delta_scale * 2 + 1 - self.delta_scale        
            for i in range(n):
                input[i] *= scale_ratio
            return input
        else:
            raise ValueError("only support ndarray or a list of ndarray")

            
# ------------------------------------------------------------------------------------
# ------------------------------------- DATASETS ------------------------------------- 
# ------------------------------------------------------------------------------------

class ScannetDepthPointGlobalDataset(Dataset):
    """Dataset for global feature pretraining with MOCO"""
    def __init__(
            self,
            split:str, 
            img_size:int=352, 
            downsample_factor:int = 8,
            num_point:int = 20000,
            num_seed:int = 1024,
            augment:bool = False,
            debug:bool = False) -> None:
        super().__init__()
        self.split = split
        self.base = ScanNetContrastBase(split)
        self.img_size = img_size
        self.downsample_factor = downsample_factor
        self.augment = augment
        self.num_point = num_point
        self.num_seed = num_seed
        self.debug = debug
        self.cropper = DepthMapCrop(diff_crop=False)
        self.pcd_aug = PointCloudAugment()
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index):
        img0, K0 = self.base[index]
        # generate two copies
        dm1, pcd1, mask1 = self.data_gen(img0, K0)
        dm2, pcd2, mask2 = self.data_gen(img0, K0)
        ret = {}
        ret["depthmap1"] = dm1
        ret["pcd1"] = pcd1
        ret["feature_map_mask1"] = mask1
        ret["depthmap2"] = dm2
        ret["pcd2"] = pcd2
        ret["feature_map_mask2"] = mask2
        return ret

    def data_gen(self, img, K):
        if self.augment:
            # rotate the original image to keep two crops aligned
            angle = np.random.rand() * 40 - 20
            img_temp = rotate_img(img, angle, center=[K[0, 2], K[0, 1]], flags=cv2.INTER_NEAREST)
        else:
            img_temp = np.copy(img)
        # crop
        img1, img2, K1, K2 = self.cropper(img_temp, K)
        # fix image size
        img1, K1 = letter_box(img1, K1, [self.img_size, self.img_size])

        if self.augment:
            if np.random.rand() > 0.5:
                img1, K1 = flip_image(img1, K1)
                img2, K2 = flip_image(img2, K2)
            bad_pixels = np.random.random(img1.shape) > 0.2
            img1 *= bad_pixels.astype(np.float32)
            # img1 = box_erase(img1, low=0.2, high=0.4)
            # img2 = box_erase(img2, low=0.2, high=0.4)
        
        pcd = depth2points(img2, K2)
        pcd = remove_zeros(pcd)
        random_idx = np.random.choice(len(pcd), self.num_point, replace= len(pcd)<self.num_point)
        pcd = pcd[random_idx, :]        # random_sample point cloud            
        if self.augment:
            pcd = self.pcd_aug(pcd)

        # generate feature map mask for 2D CNN
        img_res, K_res = downsample_img_calib(img1, K1, self.downsample_factor)
        fm_xyz = depth2points(img_res, K_res)
        fm_xyz, mask = remove_zeros(fm_xyz, return_mask=True)
        fm_xyz_zero_mask = np.nonzero(mask)[0]
        fm_xyz_fps_idx = farthest_point_sample(fm_xyz, self.num_seed)

        depthmap = img1.astype(np.float32)/MAX_DEPTH 
        return (np.expand_dims(depthmap, 0), 
                pcd.astype(np.float32),
                fm_xyz_zero_mask[fm_xyz_fps_idx].astype(np.int64))
            

class ScannetDepthPointDataset(Dataset):
    def __init__(
            self, 
            split:str, 
            img_size:int=352, 
            downsample_factor:int = 8,
            num_match = 512,
            num_point = 20000,
            num_seed = 1024,
            match_thresh = 0.05,
            diff_crop = False,
            augment = False,
            cam_conv = False,
            debug = False,
            num_pairs = 1
        ) -> None:
        self.split = split
        self.base = ScanNetContrastBase(split)
        self.img_size = img_size
        self.downsample_factor = downsample_factor
        self.num_match = num_match
        self.match_thresh = match_thresh
        self.augment = augment
        self.num_point = num_point
        self.num_seed = num_seed
        self.diff_crop = diff_crop
        self.debug = debug
        self.num_pairs = num_pairs
        self.cropper = DepthMapCrop(diff_crop)
        self.pcd_aug = PointCloudAugment()
        self.cam_conv = cam_conv
    
    def __len__(self) -> int:
        return len(self.base)
    
    def _data_gen(self, img0, K, drop_box=False) -> Dict[str, np.ndarray]:
        # make sure enough matched pairs between two views
        while (True):
            if self.augment:
                # rotate the original image to keep two crops aligned
                angle = np.random.rand() * 40 - 20
                img = rotate_img(img0, angle, center=[K[0, 2], K[0, 1]], flags=cv2.INTER_NEAREST)
            else:
                img = np.copy(img0)
            
            # crop image
            img1, img2, K1, K2 = self.cropper(img, K)
            
            # fix image size
            img1, K1 = letter_box(img1, K1, [self.img_size, self.img_size])

            if self.augment:
                if np.random.rand() > 0.5:
                    img1, K1 = flip_image(img1, K1)
                    img2, K2 = flip_image(img2, K2)
                bad_pixels = np.random.random(img1.shape) > 0.2
                img1 *= bad_pixels.astype(np.float32)
                if drop_box:
                    img1 = box_erase(img1, low=0.2, high=0.4)
                    # no need add bad pixel for img2
                    # pcd is sampled anyway
                    img2 = box_erase(img2, low=0.2, high=0.4)

            pcd = depth2points(img2, K2)      # point cloud in camera coordinate
            pcd = remove_zeros(pcd)
            
            if len(pcd) < self.num_seed * 2:
                # go to a new loop if get too less points
                continue

            random_idx = np.random.choice(len(pcd), self.num_point, replace= len(pcd)<self.num_point)
            pcd = pcd[random_idx, :]        # random_sample point cloud
            pcd_fps_idx = farthest_point_sample(pcd, self.num_seed*2)
            pcd_fps = pcd[pcd_fps_idx[:self.num_seed], :]
            
            # downsample to the scale of feature map
            img_res, K_res = downsample_img_calib(img1, K1, self.downsample_factor)
            fm_xyz = depth2points(img_res, K_res)
            fm_xyz, mask = remove_zeros(fm_xyz, return_mask=True)
            fm_xyz_zero_mask = np.nonzero(mask)[0]
            fm_xyz_fps_idx = farthest_point_sample(fm_xyz, self.num_seed)
            fm_xyz_fps = fm_xyz[fm_xyz_fps_idx]
            
            # match fnn
            nn_xyz, nn_ind = knn_flann(fm_xyz_fps, pcd_fps, k=1, parallel=False)
            nn_xyz = np.squeeze(nn_xyz, 1)
            nn_ind = np.squeeze(nn_ind, 1)
            dist_mask = np.linalg.norm(nn_xyz - fm_xyz_fps, axis=1, ord=2) < self.match_thresh
            ind1 = dist_mask.nonzero()[0]
            
            # break loop if has enough overlap (80% of num_match are unique)
            # FIXME: might slow down the training. use lower rate if necessary
            if len(ind1) > self.num_match//5*4:
                break

        match_sample = np.random.choice(len(ind1), self.num_match, replace= len(ind1)<self.num_match)
        ind3d = nn_ind[ind1[match_sample]]      # index for point cloud input
        ind2d = ind1[match_sample]              # index for depth map input

        if self.augment:
            pcd = self.pcd_aug(pcd)
            
            # jitter pcd
            # NOTE: loss is hard to optimize with jitter
            # pcd = jitter_pcd(pcd, sigma=0.01)

            # jitter depth map
            # img1 = jitter_depth(img1, sigma=0.01)
            
            # random distortion depth
            img1 = random_distort_depth(img1, delta=0.3)

            # set d <-- max_depth - d
            # change the histogram of depth
            # if np.random.rand() > 0.5:
            #    img1 = flip_depth(img1, MAX_DEPTH)

        ret = {}
        depthmap = img1.astype(np.float32)/MAX_DEPTH                        # max range = 8
        ret["depthmap"] = np.expand_dims(depthmap, 0)
        ret["pcd"] = pcd.astype(np.float32)
        ret["pcd_fps_ind"] = pcd_fps_idx.astype(np.int32)
        ret["ind_res"] = ind2d.astype(np.int64)                     # num_match
        ret["ind_fps"] = ind3d.astype(np.int64)                     # num_match
        ret["img_res"] = img_res.astype(np.float32)
        ret["feature_map_mask"] = fm_xyz_zero_mask[fm_xyz_fps_idx].astype(np.int64) 
        # used register feature map to samples

        if self.cam_conv:
            # calculate features for cam conv
            h, w = img1.shape
            cam_feat = CAMFeatureas(h, w, K1[0, 2], K1[1, 2], fx=K1[0, 0], fy=K1[1, 1])
            fs8, fs16, fs32 = cam_feat([8, 16, 32])
            ret["cam_feat_s8"] = np.moveaxis(fs8, [0, 1, 2], [1, 2, 0]).astype(np.float32)
            ret["cam_feat_s16"] = np.moveaxis(fs16, [0, 1, 2], [1, 2, 0]).astype(np.float32)
            ret["cam_feat_s32"] = np.moveaxis(fs32, [0, 1, 2], [1, 2, 0]).astype(np.float32)
        
        if self.debug:
            # for debug:
            ret["fm_fps"] = fm_xyz_fps
            ret["k1"] = K1.astype(np.float32)
            ret["ind_seed"] = pcd_fps_idx.astype(np.int64)
        return ret        
        
    def __getitem__(self, index: int) -> Dict:
        # read from base dataset
        img0, K = self.base[index]
        if self.num_pairs == 1:
            return self._data_gen(img0, np.copy(K), True)
        elif self.num_pairs == 2:
            # NOTE: drop box only for the first crop
            temp = [self._data_gen(img0, np.copy(K), True),
                    self._data_gen(img0, np.copy(K), True)]
            ret = {}
            for i in range(self.num_pairs):
                for key in temp[i]:
                    ret["{:s}{:d}".format(key, i+1)] = temp[i][key]
            return ret
        else:
            raise NotImplementedError


class ScanNetPointVoxelDataset(Dataset):
    def __init__(
            self, 
            split:str, 
            num_point = 20000,
            num_match = 512,
            num_seed = 1024,
            voxel_size = 0.05,
            match_thresh = 0.05,
            diff_crop = False,
            augment = False,
            debug = False,
            feat_dim = 1
        ) -> None:
        self.split = split
        self.base = ScanNetContrastBase(split)
        self.num_point = num_point
        self.num_match = num_match
        self.num_seed = num_seed
        self.match_thresh = match_thresh
        self.augment = augment
        self.voxel_size = voxel_size
        self.diff_crop = diff_crop
        self.debug = debug
        self.cropper = DepthMapCrop(diff_crop)
        self.pcd_aug = PointCloudAugment()
        self.feat_dim = feat_dim
    
    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict:
        # read from base dataset
        img0, K = self.base[index]    

        # make sure enough matched pairs between two views
        while (True):
            # crop image
            img1, img2, K1, K2 = self.cropper(img0, K)

            if self.augment:
                img1 = box_erase(img1, low=0.2, high=0.4)
                img2 = box_erase(img2, low=0.2, high=0.4)
            
            pcd1 = depth2points(img1, K1)
            pcd1 = remove_zeros(pcd1)
            pcd2 = depth2points(img2, K2)
            pcd2 = remove_zeros(pcd2)         
            # random sample
            random_idx = np.random.choice(len(pcd1), self.num_point, replace= len(pcd1)<self.num_point)
            pcd1 = pcd1[random_idx]
            # fps
            pcd1_fps_idx = farthest_point_sample(pcd1, self.num_seed)
            pcd1_fps = pcd1[pcd1_fps_idx, :]
            
            if self.augment:
                # only augment pcd2 and FPS points. original pcd is NOT changed!
                pcd1_fps, pcd2 = self.pcd_aug([pcd1_fps, pcd2])
            
            # voxel sample
            feat_vox = np.ones((len(pcd2), self.feat_dim))
            vox, feat_vox = ME.utils.sparse_quantize(pcd2, feat_vox, quantization_size=self.voxel_size)
            vox = vox.numpy()
            
            # match nn
            nn_xyz, nn_ind = knn_flann(pcd1_fps, vox * self.voxel_size, k=1, parallel=False)
            nn_xyz = np.squeeze(nn_xyz, 1)
            nn_ind = np.squeeze(nn_ind, 1)
            dist_mask = np.linalg.norm(nn_xyz - pcd1_fps, axis=1, ord=2) < self.match_thresh
            ind1 = dist_mask.nonzero()[0]
            
            # break loop if has enough overlap (80% of num_match are unique)
            # FIXME: might slow down the training. use lower rate if necessary
            if len(ind1) > self.num_match//5*4:
                break
            
        match_sample = np.random.choice(len(ind1), self.num_match, replace= len(ind1)<self.num_match)
        
        ind_pts = ind1[match_sample]
        ind_vox = nn_ind[ind1[match_sample]]
        
        if self.augment:
            pcd1 = self.pcd_aug(pcd1)   # NOTE: now, only augment the points input!
        
        ret = {}
        ret["pcd"] = pcd1.astype(np.float32)
        ret["vox"] = vox.astype(np.int32)
        ret["feat_vox"] = feat_vox.astype(np.float32)
        ret["ind_pts"] = ind_pts.astype(np.int64)
        ret["ind_vox"] = ind_vox.astype(np.int64)
        if self.debug:
            ret["pcd_fps"] = pcd1_fps.astype(np.float32)
            ret["ind_fps"] = pcd1_fps_idx.astype(np.int64)
        return ret


class ScanNetPointPointDataset(Dataset):
    """datset for point-point contrast"""
    def __init__(
            self, 
            split:str, 
            num_point = 20000,
            num_match = 512,
            num_seed = 1024,
            match_thresh = 0.05,
            voxel_size = 0.02,
            voxelize = False,
            diff_crop = False,
            augment = False,
            erase=True,
            debug = False
        ) -> None:
        self.split = split
        self.base = ScanNetContrastBase(split)
        self.num_point = num_point
        self.num_match = num_match
        self.num_seed = num_seed
        self.match_thresh = match_thresh
        self.augment = augment
        self.diff_crop = diff_crop
        self.debug = debug
        self.cropper = DepthMapCrop(diff_crop)
        self.pcd_aug = PointCloudAugment()
        self.voxelize = voxelize        # use voxelization as data augmentation
        self.voxel_size = voxel_size
        self.erase = erase
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index) -> Dict[str,ndarray]:
        # read from base dataset
        img0, K = self.base[index]    

        # make sure enough matched pairs between two views
        while (True):
            # crop image
            img1, img2, K1, K2 = self.cropper(img0, K)

            if self.erase:
                img1 = box_erase(img1, low=0.1, high=0.3)
                img2 = box_erase(img2, low=0.1, high=0.3)
            
            pcd1 = depth2points(img1, K1)
            pcd1 = remove_zeros(pcd1)
            pcd2 = depth2points(img2, K2)
            pcd2 = remove_zeros(pcd2)         
            # random sample
            random_idx = np.random.choice(len(pcd1), self.num_point, replace= len(pcd1)<self.num_point)
            pcd1 = pcd1[random_idx]
            # qunatize pcd2
            if self.voxelize:
                # pcd2 = ME.utils.sparse_quantize(pcd2, quantization_size=self.voxel_size)
                # pcd2 = pcd2.numpy() * self.voxel_size
                pcd2 = ME.utils.quantization.sparse_quantize(pcd2/self.voxel_size) * self.voxel_size
                pcd2 = pcd2.numpy()
            random_idx = np.random.choice(len(pcd2), self.num_point, replace= len(pcd2)<self.num_point)
            pcd2 = pcd2[random_idx]            
            # fps
            # NOTE: pointnet2 backbone use this fps index directly!
            # CUDA and C++ implementation sometimes generate different result!
            pcd1_fps_idx = farthest_point_sample(pcd1, self.num_seed*2)
            pcd1_fps = pcd1[pcd1_fps_idx[:self.num_seed], :]
            pcd2_fps_idx = farthest_point_sample(pcd2, self.num_seed*2)
            pcd2_fps = pcd2[pcd2_fps_idx[:self.num_seed], :]
            
            # match nn
            nn_xyz, nn_ind = knn_flann(pcd1_fps, pcd2_fps, k=1, parallel=False)
            nn_xyz = np.squeeze(nn_xyz, 1)
            nn_ind = np.squeeze(nn_ind, 1)
            dist_mask = np.linalg.norm(nn_xyz - pcd1_fps, axis=1, ord=2) < self.match_thresh
            ind1 = dist_mask.nonzero()[0]
            
            # break loop if has enough overlap (80% of num_match are unique)
            # FIXME: might slow down the training. use lower rate if necessary
            if len(ind1) > self.num_match//5*4:
                break
        
        match_sample = np.random.choice(len(ind1), self.num_match, replace= len(ind1)<self.num_match)
        temp = ind1[match_sample]
        ind2 = nn_ind[ind1[match_sample]]
        ind1 = temp
        
        if self.augment:
            pcd1 = self.pcd_aug(pcd1)
            pcd2 = self.pcd_aug(pcd2)
        
        ret = {}
        ret["pcd1"] = pcd1.astype(np.float32)
        ret["pcd2"] = pcd2.astype(np.float32)
        ret["ind1"] = ind1.astype(np.int64)
        ret["ind2"] = ind2.astype(np.int64)
        ret["fps_ind1"] = pcd1_fps_idx.astype(np.int32)
        ret["fps_ind2"] = pcd2_fps_idx.astype(np.int32)
        return ret        


class ScanNetDepthVoxelDataset(Dataset):
    """suppose output of 3D CNN has stride = 1, 
    output of 2D CNN has stride = 2"""
    def __init__(
        self,
        split : str, 
        img_size : int = 352,
        num_match : int = 2048,
        match_thresh : float = 0.04,
        voxel_size : float = 0.04,
        use_rgb : bool = True,
        debug : bool = False,
        num_pairs : int = 1):
        super().__init__()
        self.split = split
        self.use_rgb = use_rgb
        self.base = ScanNetContrastBase(split, use_rbg=use_rgb)
        self.img_size = img_size
        self.num_match = num_match
        self.match_thresh = match_thresh
        self.debug = debug
        self.num_pairs = num_pairs
        self.pcd_aug = PointCloudAugment()
        self.voxel_size = voxel_size
        self.transform = tvt.Compose([
            tvt.RandomApply([tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            # tvt.RandomSolarize(120),
            tvt.RandomApply([tvt.GaussianBlur(5)]),
            tvt.RandomGrayscale(0.2),
        ])
    
    def __len__(self) -> int:
        return len(self.base)
    
    def _data_gen(self, rgbd : RGBDImage) -> Dict[str, np.ndarray]:
        angle = np.random.rand() * 40 - 20
        rgbd.rotate(angle)
        rgbd.random_crop(0.7, 0.9)
        # for depth map input
        rgbd1 = deepcopy(rgbd)
        rgbd1.transform_rgb(self.transform)
        # for voxel input
        rgbd2 = deepcopy(rgbd)
        rgbd2.transform_rgb(self.transform)
        
        rgbd1.letter_box([self.img_size, self.img_size])
        if np.random.rand() > 0.5:
            rgbd1.flip_h()
            rgbd2.flip_h()
        rgbd1.random_box_erase(0.2, 0.4)
        rgbd2.random_box_erase(0.2, 0.4)
        # add bad pixels to depth map
        drop = np.random.random(rgbd1.depth.shape) > 0.2
        rgbd1.depth = rgbd1.depth * drop
        # downsample depth map:
        ds2, ks2 = rgbd1.downsampled_depth(2)
        imgs2 = RGBDImage(rgb=None, depth=ds2, intrinsic=ks2)
        # convert to point cloud
        pcd1, zero_mask = imgs2.to_pointcloud(True)
        pcd2 = rgbd2.to_pointcloud(False)
        xyz1 = pcd1[:, :3]
        xyz2 = pcd2[:, :3]
        # randomly translate point cloud. for voxel based net
        rand_trans = np.random.rand(1, 3) * self.voxel_size * 20
        xyz1 = xyz1 + rand_trans
        xyz2 = xyz2 + rand_trans
        # augment point cloud
        xyz1, xyz2 = self.pcd_aug([xyz1, xyz2])
        pcd1[:, :3] = xyz1
        pcd2[:, :3] = xyz2
        if self.use_rgb:
            feats = pcd2[:, 3:]
        else:
            feats = np.ones((len(xyz1), 1))
        vox = np.floor(xyz2/self.voxel_size).astype(np.int32)
        index, _ = ME.utils.quantization.quantize(vox)
        vox = vox[index]
        feats = feats[index]
        # match nn
        nn_xyz, nn_ind = knn_flann(xyz1, vox * self.voxel_size, 1, False)       
        nn_xyz = np.squeeze(nn_xyz, 1)
        nn_ind = np.squeeze(nn_ind, 1)
        dist_mask = np.linalg.norm(nn_xyz - xyz1, axis=1, ord=2) < self.match_thresh
        ind1 = dist_mask.nonzero()[0]        
        
        match_sample = np.random.choice(len(ind1), self.num_match, replace= len(ind1)<self.num_match)
        ind_depthmap = zero_mask[ind1[match_sample]]
        ind_vox = nn_ind[ind1[match_sample]]
        
        depth, rgb = rgbd1.depth, rgbd1.rgb
        ret = {}
        ret["coords"] = vox
        ret["feats"] = feats.astype(np.float32)
        ret["ind_depthmap"] = ind_depthmap.astype(np.int64)
        ret["ind_vox"] = ind_vox.astype(np.int64)
        depth = np.expand_dims(depth, 0)
        ret["depthmap"] = depth.astype(np.float32) / MAX_DEPTH
        rgb = np.transpose(rgb, (2, 0, 1))
        ret["rgb"] = rgb.astype(np.float32)
        if self.debug:
            ret["pcd1"] = pcd1
            ret["pcd2"] = pcd2
            ret["ind_pts"] = ind1[match_sample]
        return ret
        
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        if self.use_rgb: 
            depth, rgb, intr = self.base[index]
            rgb = rgb / 255.
        else:
            depth, intr = self.base[index]
            rgb = None

        rgbd = RGBDImage(rgb, depth, intr)
        
        if self.num_pairs == 1:
            return self._data_gen(deepcopy(rgbd))
        elif self.num_pairs == 2:
            # NOTE: drop box only for the first crop
            temp = [self._data_gen(deepcopy(rgbd)),
                    self._data_gen(deepcopy(rgbd))]
            ret = {}
            for i in range(self.num_pairs):
                for key in temp[i]:
                    ret["{:s}{:d}".format(key, i+1)] = temp[i][key]
            return ret
        else:
            raise NotImplementedError        

class ScanNetImagePointDataset(Dataset):
    def __init__(
        self,
        split : str, 
        img_size : int = 352,
        num_match : int = 512,
        match_thresh : float = 0.05,
        num_point : int = 20000,
        downsample_factor : int = 8,
        num_seed : int = 1024,
        debug : bool = False,
        num_pairs : int = 1):
        super().__init__()
        self.split = split
        self.base = ScanNetContrastBase(split, use_rbg=True)
        self.img_size = img_size
        self.num_match = num_match
        self.match_thresh = match_thresh
        self.debug = debug
        self.num_pairs = num_pairs
        self.pcd_aug = PointCloudAugment()
        self.downsample_factor = downsample_factor
        self.num_point = num_point
        self.num_seed = num_seed
        self.transform = tvt.Compose([
            tvt.RandomApply([tvt.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            # tvt.RandomSolarize(120),
            tvt.RandomApply([tvt.GaussianBlur(5)]),
            tvt.RandomGrayscale(0.2),
        ])
    
    def __len__(self) -> int:
        return len(self.base)
    
    def _data_gen(self, rgbd_0 : RGBDImage) -> Dict[str, np.ndarray]:
        while (True):
            rgbd = deepcopy(rgbd_0)
            angle = np.random.rand() * 40 - 20
            rgbd.rotate(angle)
            rgbd.random_crop(0.7, 0.9)
            # for image input
            rgbd1 = deepcopy(rgbd)
            rgbd1.transform_rgb(self.transform)
            # for point cloud input
            rgbd2 = deepcopy(rgbd)
            
            rgbd1.letter_box([self.img_size, self.img_size])
            if np.random.rand() > 0.5:
                rgbd1.flip_h()
                rgbd2.flip_h()
            rgbd1.random_box_erase(0.2, 0.3)
            rgbd2.random_box_erase(0.2, 0.3)
            # add bad pixels to depth map
            drop = np.random.random(rgbd1.depth.shape) > 0.2
            rgbd1.depth = rgbd1.depth * drop
            
            # build point cloud
            pcd = rgbd2.to_pointcloud(False)[:, :3]
            random_idx = np.random.choice(len(pcd), self.num_point, replace= len(pcd)<self.num_point)
            pcd = pcd[random_idx, :]        # random_sample point cloud
            
            if len(pcd) < self.num_seed * 2:
                # go to a new loop if get too less points
                continue            
            
            # FPS
            pcd_fps_idx = farthest_point_sample(pcd, self.num_seed*2)
            pcd_fps = pcd[pcd_fps_idx[:self.num_seed], :]
            
            # downsample depth map:
            ds2, ks2 = rgbd1.downsampled_depth(self.downsample_factor)
            img_scaled = RGBDImage(rgb=None, depth=ds2, intrinsic=ks2)
            
            # convert scaled image to point cloud
            fm_xyz, fm_xyz_zero_mask = img_scaled.to_pointcloud(True)
            
            fm_xyz_fps_idx = farthest_point_sample(fm_xyz, self.num_seed)
            fm_xyz_fps = fm_xyz[fm_xyz_fps_idx]
            
            # match fnn
            nn_xyz, nn_ind = knn_flann(fm_xyz_fps, pcd_fps, k=1, parallel=False)
            nn_xyz = np.squeeze(nn_xyz, 1)
            nn_ind = np.squeeze(nn_ind, 1)
            dist_mask = np.linalg.norm(nn_xyz - fm_xyz_fps, axis=1, ord=2) < self.match_thresh
            ind1 = dist_mask.nonzero()[0]

            # FIXME: might slow down the training. use lower rate if necessary
            if len(ind1) > self.num_match//2:
                break


        match_sample = np.random.choice(len(ind1), self.num_match, replace= len(ind1)<self.num_match)
        ind3d = nn_ind[ind1[match_sample]]      # index for point cloud input
        ind2d = ind1[match_sample]              # index for depth map input
        
        pcd = self.pcd_aug(pcd)
        
        
        ret = {}
        rgb = np.zeros((3, self.img_size, self.img_size))
        rgb[0, ...] = (rgbd1.rgb[:, :, 0] - 0.485) / 0.229
        rgb[1, ...] = (rgbd1.rgb[:, :, 1] - 0.456) / 0.224
        rgb[2, ...] = (rgbd1.rgb[:, :, 2] - 0.406) / 0.225
        ret["rgb"] = rgb.astype(np.float32)
        ret["pcd"] = pcd.astype(np.float32)
        ret["pcd_fps_ind"] = pcd_fps_idx.astype(np.int32)
        ret["ind_res"] = ind2d.astype(np.int64)                     # num_match
        ret["ind_fps"] = ind3d.astype(np.int64)                     # num_match
        ret["feature_map_mask"] = fm_xyz_zero_mask[fm_xyz_fps_idx].astype(np.int64)
        return ret
        
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        depth, rgb, intr = self.base[index]
        rgb = rgb / 255.

        rgbd = RGBDImage(rgb, depth, intr)
        
        if self.num_pairs == 1:
            return self._data_gen(rgbd)
        elif self.num_pairs == 2:
            # NOTE: drop box only for the first crop
            temp = [self._data_gen(rgbd),
                    self._data_gen(rgbd)]
            ret = {}
            for i in range(self.num_pairs):
                for key in temp[i]:
                    ret["{:s}{:d}".format(key, i+1)] = temp[i][key]
            return ret
        else:
            raise NotImplementedError        


# ------------------------------------------------------------------------------------
# ------------------------------------ TEST CASES ------------------------------------ 
# ------------------------------------------------------------------------------------

def test_point_point():
    ds = ScanNetPointPointDataset("train", voxelize=True, augment=True, debug=True)
    ret = ds[np.random.randint(0, len(ds))]
    pts1 = ret["pcd1"]
    pts2 = ret["pcd2"]
    fps1 = ret["fps_ind1"]
    fps2 = ret["fps_ind2"]
    ind1 = ret["ind1"]
    ind2 = ret["ind2"]
    pcd1 = create_pcd(pts1[fps1], color=[1, 0, 0])
    pcd2 = create_pcd(pts2[fps2], color=[0, 0, 1])
    lineset = create_offset(pts1[fps1[:1024]][ind1], pts2[fps2[:1024]][ind2])
    o3d.visualization.draw_geometries([pcd1, pcd2, lineset])


def test_point_voxel():
    """validate the data augmentation"""
    ds = ScanNetPointVoxelDataset("train", augment=True, debug=True)
    ret = ds[np.random.randint(0, len(ds))]
    pcd = ret["pcd"]
    ind_fps = ret["ind_fps"]
    vox = ret["vox"] * ds.voxel_size
    ind_pts = ret["ind_pts"]
    ind_vox = ret["ind_vox"]
    print(pcd.shape)
    pcd1 = create_pcd(pcd[ind_fps], [1, 0, 0])
    pcd2 = create_pcd(vox)
    lineset = create_offset(pcd[ind_fps][ind_pts], vox[ind_vox])
    o3d.visualization.draw_geometries([pcd1, pcd2, lineset])


def test_point_voxel2():
    """validate decomposition of sparse tensor"""
    sys.path.append(os.path.dirname(BASE_DIR))    
    from model.cnn3d.minkunet import MinkUNet34C
    
    ds = ScanNetPointVoxelDataset("train", augment=True, debug=True)
    ret = ds[np.random.randint(0, len(ds))]
    pcd = ret["pcd"]
    ind_fps = ret["ind_fps"]
    vox = ret["vox"]
    ind_pts = ret["ind_pts"]
    ind_vox = ret["ind_vox"]

    feat_vox = ret["feat_vox"]
    coords, feats = ME.utils.sparse_collate(coords=[vox, vox], feats=[feat_vox, feat_vox])
    sin = ME.SparseTensor(feats.float(), coords.int())
    net = MinkUNet34C(1, 256)
    sout = net(sin)
    coords_out = sout.coordinates
    
    # second minibatch (bacth_index = 1)
    c = coords_out[coords_out[:, 0] == 1, 1:]
    c = ds.voxel_size * c
    c = c.numpy()

    pcd1 = create_pcd(pcd[ind_fps], [1, 0, 0])
    pcd2 = create_pcd(c)
    lineset = create_offset(pcd[ind_fps][ind_pts], c[ind_vox])
    o3d.visualization.draw_geometries([pcd1, pcd2, lineset])
    

def test_depth_point():
    ds = ScannetDepthPointDataset("train", augment=True, diff_crop=False, cam_conv=True, debug=True)
    ret = ds[np.random.randint(0, len(ds))]
    pcd1 = ret["fm_fps"]
    pcd2 = ret["pcd"]
    depth = ret["depthmap"][0] * 8
    k1 = ret["k1"]
    idx_res = ret["ind_res"]
    idx_fps = ret["ind_fps"]
    idx_seed = ret["ind_seed"]
    mask = ret["feature_map_mask"]
    
    depth_small, k1 = downsample_img_calib(depth, k1, 8)
    pcd3 = depth2points(depth_small, k1)
    pcd3 = pcd3[mask, :][idx_res, :]
    lines = create_offset(pcd3, pcd2[idx_seed, :][idx_fps, :])
    
    pcd1 = create_pcd(pcd1, [0, 0, 1])
    pcd2 = create_pcd(pcd2[idx_seed, :], [1, 0, 0])
    o3d.visualization.draw_geometries([pcd1, pcd2, lines])
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(depth)
    plt.show()
    
    fs8 = ret["cam_feat_s8"]
    fs16 = ret["cam_feat_s16"]
    print(fs8.shape)
    print(fs16.shape)


def test_image_point():
    ds = ScanNetImagePointDataset("train")
    ret = ds[np.random.randint(0, len(ds))]
    for key in ret:
        print(ret[key].shape)


def test_num_pairs():
    print("num_pairs == 1")
    ds = ScannetDepthPointDataset(split="train", augment=True, num_pairs=1)
    ret = ds[0]
    for key in ret:
        print(key, ret[key].shape)
    print("+"*20)
    print("num_pairs == 2")
    ds = ScannetDepthPointDataset(split="train", augment=True, num_pairs=2)
    ret = ds[0]
    for key in ret:
        print(key, ret[key].shape)


def test_base():
    from data_utils import create_frame
    from rgbd_image import RGBDImage
    import matplotlib.pyplot as plt
    ds = ScanNetContrastBase("train", True)
    d, rgb, intri = ds[100]
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(d)
    plt.subplot(1, 2, 2)
    plt.imshow(rgb)
    plt.show()
    
    rgbd = RGBDImage(rgb/255, d, intri)
    pts = rgbd.to_pointcloud()
    frame = create_frame()
    pcd = create_pcd(pts[:, :3], pts[:, 3:])
    o3d.visualization.draw_geometries([pcd, frame])


def test_depth_voxel():
    from data_utils import create_frame, create_pcd, create_offset
    import matplotlib.pyplot as plt     
    ds = ScanNetDepthVoxelDataset("train", debug=True)
    ret = ds[0]
    depth = ret["depthmap"]
    rgb = ret["rgb"]
    p1 = ret["pcd1"]
    vs = ds.voxel_size
    for key in ret:
        print(key, ret[key].shape)

    print(ret["ind_vox"][:10])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(depth[0, ...])
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(rgb, (1, 2, 0)))
    plt.show()
    
    pcd1 = create_pcd(p1[:, :3])
    # NOTE: translate the point cloud to create offset
    pcd2 = create_pcd(ret["coords"] * vs + 1, ret["feats"])
    lineset = create_offset(
        p1[ret["ind_pts"], :3], 
        ret["coords"][ret["ind_vox"]] * vs + 1, 
        color=[1, 0, 0])
    frame = create_frame()
    o3d.visualization.draw_geometries([pcd1, pcd2, lineset, frame])
