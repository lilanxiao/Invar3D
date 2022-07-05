import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

def _flip(pts_o):
    # xyz: right, front, up
    # to: right, down, front
    pts = deepcopy(pts_o)
    pts = pts[:, [0, 2, 1]]
    pts[:, 1] = - pts[:, 1]
    return pts

def read_scannet_instri(path:str):
    with open(path) as f:
        lines = f.read().splitlines()
    temp = None
    for l in lines:
        if "m_calibrationDepthIntrinsic" in l:
            temp = l
    num = temp.split("=")[1]    # take numbers after =
    array = [float(x) for x in num.split(" ") if len(x)>0]
    array = np.array(array)
    array = np.reshape(array, (4, 4))
    return array[:3, :3]


def _png2depth(path, true_depth=True):
    """read a png image and convert it to a depth map

    Arguments:
        path {str} -- path of a image
        true_depth {bool} -- convert unit16 to true depth value in meter as type float

    Returns:
        [numpy array] -- depth map
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if true_depth:
        img = img.astype(np.float32)/1000. # to real depth
    # img[img>8]=8
    return img


def depth2points(depth_map, K):
    cx = K[0,2]
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]
    H,W = depth_map.shape
    x,y = np.meshgrid(np.arange(W), np.arange(H))
    x3 = np.ravel((x-cx) * depth_map / fx)
    y3 = np.ravel((y-cy) * depth_map / fy)
    z3 = np.ravel(depth_map)
    points = np.concatenate([np.expand_dims(x3,-1),np.expand_dims(z3,-1),np.expand_dims(-y3,-1)],-1)
    return points


def _get_pose(path:str):
    """get pose from txt file

    Args:
        path (str): [description]

    Returns:
        numpy.ndarray: rotation
        numpy.ndarray: translation
    """
    # NOTE: scannt has some frame with invalid pose (-inf)
    data = np.loadtxt(path)
    return data[0:3, 0:3], data[0:3, 3]


def remove_zeros(point_cloud:np.ndarray, threshhold:float=0.1, return_mask:bool=False):
    mask = point_cloud[:, 1]>threshhold
    if return_mask:
        return point_cloud[mask, :], mask
    else:
        return point_cloud[mask, :]


def pose2extr(rot):
    """convert global pose in a whole scene into extrinsic
    """
    r0 = R.from_matrix(rot)
    ang = r0.as_euler("xyz")
    ang[0] = np.pi/2  + ang[0]      # flip axis
    ang[2] = 0.                     # remove rotation around z
    r1 = R.from_euler("xyz", ang)
    return r1.as_matrix()


def lr_mirror(rot):
    r0 = R.from_matrix(rot)
    ang = r0.as_euler("xyz")
    ang[1] = - ang[1]
    ang[2] = - ang[2]
    r1 = R.from_euler("xyz", ang)
    return r1.as_matrix()


def create_pcd(array, color=[0.5, 0.5, 0.5]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    if color is not None:
        if len(color)==3:
            pcd.paint_uniform_color(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def create_offset(end1, end2, color=[0,0,0]):
    assert end1.shape == end2.shape, "input matices should have the same size!"
    N = end1.shape[0]
    arr = np.arange(N)
    relation = np.stack([arr, arr+N], axis=1)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.concatenate([end1, end2], axis=0))
    lineset.lines = o3d.utility.Vector2iVector(relation)
    lineset.paint_uniform_color(color)
    return lineset


def create_frame():
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    return frame


def crop_depthmap(map:np.ndarray, x1:int, y1:int, x2:int, y2:int)->np.ndarray:
    """crop depthmap with zero padding

    Args:
        map (np.ndarray): [description]
        x1 (int): left above corner
        y1 (int): left above corner
        x2 (int): right bottom corner
        y2 (int): right bottom corner

    Returns:
        np.ndarray: (H, W)
    """
    H, W = map.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, W)
    y2 = min(y2, H)
    ret = np.copy(map)
    ret[0:y1, :] = 0
    ret[y2:H, :] = 0
    ret[:, 0:x1] = 0
    ret[:, x2:W] = 0
    return ret


def rotate_img(image, angle, center = None, scale = 1.0, flags = cv2.INTER_CUBIC):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags, flags)
    return rotated


def downsample_img_calib(img, K, factor):
    H, W = img.shape
    K_res = K / factor
    K_res[2,2] = 1.
    # NOTE: careful with order of H and W
    # should use nearst neighbor interpolation because some pixels have value zero
    img_res = cv2.resize(img, dsize=(int(np.ceil(W/factor)), int(np.ceil(H/factor))), interpolation=cv2.INTER_NEAREST)
    return img_res, K_res


def letter_box(img:np.ndarray, K:np.ndarray, target_size, flags=cv2.INTER_NEAREST):
    H, W = img.shape[0:2]
    th, tw = target_size
    scale = min(th/H, tw/W)
    nh = int(H * scale)
    nw = int(W * scale)
    img_res = cv2.resize(img, (nw, nh), interpolation=flags)
    if len(img.shape) == 2:
        new_img = np.full((th, tw), 0, dtype=img.dtype)
    else:
        new_img = np.full((th, tw, 3), 0, dtype=img.dtype)
    new_img[(th - nh)//2 : (th - nh)//2 + nh, (tw - nw) // 2 : (tw - nw) // 2 + nw] = img_res

    # resize 
    K_res = K * scale
    K_res[2, 2] = 1.
    # padding
    K_res[0, 2] += (tw - nw)//2     # cx
    K_res[1, 2] += (th - nh)//2     # cy
    return new_img, K_res


from numpy import ndarray
from typing import Tuple
def random_crop(img:ndarray, K:ndarray, min_ratio=0.7, max_ratio=0.9) -> Tuple[ndarray, ndarray]:
    """random crop a region and update the instrinsic. 

    Args:
        img (ndarray): input image
        K (ndarray): intrinsic
        min_ratio (float, optional): minimal crop ratio. Defaults to 0.7.
        max_ratio (float, optional): maximal crop ratio. Defaults to 0.9.

    Returns:
        ndarray: cropped image
        ndarray: new intrinsic
    """
    ratio = np.random.rand() * (max_ratio - min_ratio) + min_ratio
    H, W = img.shape[0], img.shape[1]
    nh = int(H*ratio)

    ratio = np.random.rand() * (max_ratio - min_ratio) + min_ratio    
    nw = int(W*ratio)
    
    up = int((H - nh) * np.random.rand())
    left = int((W - nw) * np.random.rand())
    
    new_img = img[up:up+nh, left:left+nw]
    new_K = deepcopy(K)
    new_K[0, 2] -= left
    new_K[1, 2] -= up
    return new_img, new_K


def flip_image(img:ndarray, K:ndarray) -> Tuple[ndarray, ndarray]:
    """left right flip the image"""
    W = img.shape[1]
    ind = np.arange(W-1, -1, -1)
    img = img[:, ind]
    # intrinsic
    K_new = deepcopy(K)
    K_new[0, 2] = W - K_new[0, 2]
    return img, K


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s,  0,  c]])


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c,  -s],
                     [0,  s,  c]])


def jitter_depth(img:ndarray, sigma=0.01, percentage=0.3)->ndarray:
    """add normal distributed jitter to depth maps

    Args:
        img (ndarray): depth map
        sigma (float, optional): sigma in meter. Defaults to 0.01.
        percentage: percentage of pixels to jitter

    Returns:
        ndarray: image with jitter
    """
    non_zeros = (img > 0).astype(img.dtype)
    noise = np.random.randn(*img.shape) * sigma
    mask = (np.random.rand(*img.shape) < percentage).astype(np.float32)
    noise *= non_zeros
    return img + noise * mask


def jitter_pcd(pcd:ndarray, sigma=0.01, percentage=0.3)->ndarray:
    """jitter point cloud

    Args:
        pcd (ndarray): point cloud
        sigma (float, optional): sigma in meter. Defaults to 0.01.
        percentage: percentage of points to jitter

    Returns:
        ndarray: jittered pcd
    """
    noise = np.random.randn(*pcd.shape) * sigma
    mask = (np.random.rand(*pcd.shape) < percentage).astype(np.float32)
    return pcd + noise * mask


def box_erase(img:ndarray, low=0.3, high=0.5)->ndarray:
    h, w = img.shape[0], img.shape[1]
    cy = int(np.floor(np.random.rand() * h))
    cx = int(np.floor(np.random.rand() * w))
    ch = h * ( np.random.rand() * (high - low) + low ) / 2
    cw = w * ( np.random.rand() * (high - low) + low ) / 2
    ch = int(ch)
    cw = int(cw)
    new_img = np.copy(img)
    new_img[max(0, cy-ch):min(h, cy+ch), max(0, cx-cw):min(w, cx+cw)] = 0.
    return new_img


def flip_depth(arr : ndarray, max_depth : float) -> ndarray:
    zero_mask = (arr > 1e-2).astype(arr.dtype)
    rarr = max_depth - arr
    return zero_mask * rarr


def random_distort_depth(arr : np.ndarray, delta : float):
    """randomly distort the depth map by scaling

    Args:
        arr (np.ndarray): depth map
        delta (float): start and end of scaling factor [1-delta, 1+delta]

    Returns:
        ndarray: distorted depth map 
    """
    start = np.random.rand() * 2 * delta - delta + 1
    stop = np.random.rand() * 2 * delta - delta + 1
    arr = distort_depth(arr, True, start, stop)
    start = np.random.rand() * 2 * delta - delta + 1
    stop = np.random.rand() * 2 * delta - delta + 1
    arr = distort_depth(arr, False, start, stop)
    return arr


def distort_depth(depth : np.ndarray, x : bool, start : float, stop : float):
    """apply linear distortion to depth map

    Args:
        depth (np.ndarray): depth map
        x (bool): to x-axis or y-axis
        start (float): start 
        stop (float): stop

    Returns:
        ndarray: distorted depth map 
    """
    h, w = depth.shape[0], depth.shape[1]
    if x:
        arr = np.linspace(start, stop, w).astype(depth.dtype)
        arr = np.expand_dims(arr, 0)
        arr = np.repeat(arr, h, 0)
    else:
        arr = np.linspace(start, stop, h).astype(depth.dtype)
        arr = np.expand_dims(arr, 1)
        arr = np.repeat(arr, w, 1)
    return depth * arr


if __name__ == "__main__":
    img = np.random.rand(244, 244)
    K = np.random.rand(3, 3)
    new_img, new_K = letter_box(img, K, [200, 200])
    print(new_img.shape)

    new_img, new_K = random_crop(img, K)
    print(new_img.shape)
    
    import matplotlib.pyplot as plt
    img = np.random.rand(200, 200)
    img = box_erase(img)
    plt.figure()
    plt.imshow(img)
    plt.show()