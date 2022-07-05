# encode camera calibration (intrinsic) as feature maps
# https://github.com/jmfacil/camconvs

import numpy as np
import cv2
from typing import Tuple, List

class CAMFeatureas(object):
    def __init__(self,
                h:float,
                w:float,
                cx:float,
                cy:float,
                fx:float,
                fy:float,
                coord_maps:bool=False,
                centered_coord_maps:bool=True,
                norm_coord_maps:bool=True,
                r_maps:bool=False,
                border_dist_maps:bool=False,
                fov_maps:bool=True,
                scale:float=320.
                ) -> None:
        """
        Args:
            h (float): height of image
            w (float): width of image
            cx (float): center x
            cy (float): center y
            fx (float): focal length
            fy (float): focal length
            coord_maps (bool, optional): use row coordinate maps. Defaults to False.
            centered_coord_maps (bool, optional): use centered coordinate maps. Defaults to True.
            norm_coord_maps (bool, optional): use normalize coodinate maps. Defaults to True.
            r_maps (bool, optional): use radius maps. Defaults to False.
            border_dist_maps (bool, optional): use distance to border. Defaults to False.
            fov_maps (bool, optional): use field of view maps. Defaults to True.
            scale (float, optional): scale factor of coordinates. Defaults to 320..
        """
        self.h = h
        self.w = w
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.coord_maps = coord_maps
        self.centered_coord_maps = centered_coord_maps
        self.norm_coord_maps = norm_coord_maps
        self.r_maps = r_maps
        self.border_dist_maps = border_dist_maps
        self.fov_maps = fov_maps
        self.scale = scale
    
    def channels(self) -> int:
        c = ( 2*int(self.coord_maps) 
            + 2*int(self.centered_coord_maps)
            + 2*int(self.norm_coord_maps)
            + 1*int(self.r_maps)
            + 4*int(self.border_dist_maps)
            + 2*int(self.fov_maps))
        return c
    
    def _define_coord_channels(self) -> Tuple[np.ndarray, np.ndarray]:
        x_channels = np.arange(0, self.w)
        x_channels = np.expand_dims(x_channels, 0)
        x_channels = np.repeat(x_channels, self.h, axis = 0)
        y_channels = np.arange(0, self.h)
        y_channels = np.expand_dims(y_channels, 1)
        y_channels = np.repeat(y_channels, self.w, axis = 1)
        return x_channels, y_channels
    
    def features(self) -> np.ndarray:
        ret = []
        x_channels, y_channels = self._define_coord_channels()
        if self.coord_maps:
            ret = ret + [x_channels, y_channels]
        
        if self.centered_coord_maps or self.fov_maps or self.r_maps:
            cent_x_channels = x_channels - self.cx
            cent_y_channels = y_channels - self.cy
            if self.centered_coord_maps:
                ret += [cent_x_channels/self.scale, cent_y_channels/self.scale]
            if self.fov_maps:
                fov_x = np.arctan(cent_x_channels/self.fx)
                fov_y = np.arctan(cent_y_channels/self.fy)
                ret += [fov_x, fov_y]
            if self.r_maps:
                radius = np.sqrt((cent_x_channels/self.scale)**2 + (cent_y_channels/self.scale)**2)
                ret.append(radius)
        
        if self.norm_coord_maps:
            norm_x_channels = x_channels/(self.w - 1) * 2. - 1.
            norm_y_channels = y_channels/(self.h - 1) * 2. - 1.
            ret += [norm_x_channels, norm_y_channels]
        
        if self.border_dist_maps:
            lef = x_channels
            rig = self.w - x_channels - 1
            top = y_channels
            bot = self.h - y_channels - 1
            ret += [lef, rig, top, bot]
        
        features = np.stack(ret, 2)
        return features

    def __call__(self, stride:List[int]=None) -> List[np.ndarray]:
        """return feature maps. if stride is int or None, return ndarray.
        if stride is a list, return a list of ndarray. Channels are in the last axis.

        Args:
            stride (List[int], optional): stride of feature maps. Defaults to None.

        Raises:
            ValueError: stride is not None or int or List[int]

        Returns:
            List[np.ndarray] or ndarray: feature map
        """
        feat = self.features()
        h, w, _ = feat.shape        
        
        if stride is None:
            return feat
        elif isinstance(stride, int):
            return cv2.resize(feat, (w//stride, h//stride), interpolation=cv2.INTER_LINEAR)
        elif isinstance(stride, List):
            ret = [cv2.resize(feat, (w//s, h//s), interpolation=cv2.INTER_LINEAR) for s in stride]
            return ret
        else:
            raise ValueError("unsupport type")
        
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cam = CAMFeatureas(h=320, w=480, cx=220, cy=180, fx=500, fy=500)
    print(cam.channels())
    
    # check shape at each stride
    stride = [1, 2, 4, 8]
    ret = cam(stride)
    for s,f in zip(stride, ret):
        print("stride: {:d}, size:".format(s), f.shape)
    
    # visualization
    plt.figure()
    for i in range(cam.channels()):
        plt.subplot(2, 3, i+1)
        plt.imshow(ret[0][..., i])
    plt.show()
    
    # swap axis to fit pytorch
    cam_swap = np.moveaxis(ret[0], [0, 1, 2], [1, 2, 0])
    print(cam_swap.shape)
