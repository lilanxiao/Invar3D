import MinkowskiEngine as ME
import torch
import numpy as np
from torch import Tensor

@torch.no_grad()
def nn_search(x1:Tensor, x2:Tensor):
    """find nearest neighbor of each point

    Args:
        x1 (Tensor): (n1, d)
        x2 (Tensor): (n2, d)

    Returns:
        dist1:  distance (n1, )
        dist2:  distance (n1, )
        idx1:   index (n2, )
        idx2:   index (n2, )
    """
    dist = l2_distance(x1, x2)
    dist1, idx1 = torch.min(dist, dim=1)    # (n1, )
    dist2, idx2 = torch.min(dist, dim=0)    # (n2, )
    return dist1, dist2, idx1, idx2


def parse_sparse_tensor(x:ME.SparseTensor, voxel_size:float):
    """decompose sparse tensor, restore the coordinates

    Args:
        x (ME.SparseTensor): sparse tensor
        voxel_size (float): size of voxel at this scale level

    Returns:
        coords: list of coordinate tensor
        features: list of feature tensor
    """
    coords, features = x.decomposed_coordinates_and_features
    # restore the sparse coordinate
    coords = [c*voxel_size for c in coords]
    return coords, features


def l2_distance(x1:Tensor, x2:Tensor) -> Tensor:
    """calcualte pairwise distance of x1 and x2.
    similiar to torch.cdist()

    Args:
        x1 (Tensor): (n1, d)
        x2 (Tensor): (n2, d)

    Returns:
        Tensor: (n1, n2)
    """
    dist = -2 * torch.matmul(x1, x2.transpose(0, 1))
    dist += torch.sum(x1 * x1, -1)[:, None]
    dist += torch.sum(x2 * x2, -1)[None, :]
    return dist    


def match_pcd_voxel(xyz_batch: Tensor, 
                    feat_batch: Tensor, 
                    sparse_tensor: ME.SparseTensor, 
                    voxel_size: float,
                    num_match:int,
                    thresh:float=0.05):
    """match point cloud and voxel

    Args:
        xyz_batch (Tensor): (B, N, 3)
        feat_batch (Tensor): (B, C, N)
        sparse_tensor (ME.SparseTensor): sparse tensor
        voxel_size (float): voxel size at this scale level
        num_match (int): number of matched
        thresh (float, optional): threshold of distance. Defaults to 0.05.

    Returns:
        result_p: (B, C, num_match), point cloud features
        result_v: (B, C, num_match), voxel features
    """
    B, C, N = feat_batch.size()
    # decompose sparse tensor
    coords, feats_v = parse_sparse_tensor(sparse_tensor, voxel_size)
    
    result_p = torch.ones(B, C, num_match).to(xyz_batch.device)
    result_v = torch.ones(B, C, num_match).to(xyz_batch.device)
    
    for i in range(B):
        xyz = xyz_batch[i, ...]
        feat_p = feat_batch[i, ...]
        coord = coords[i, ...]
        feat_v = feats_v[i, ...].transpose(0, 1)    # (N, C) to (C, N)
        # search nearest neighbor under threshold
        dist, idx, _, _ = nn_search(xyz, coord)
        close_match = torch.nonzero(dist < thresh, as_tuple=True)
        num_close_macth = close_match.size(0)
        
        # handle very few match: directly abandon this mini-batch        
        if num_close_macth < num_match//2:
            continue

        # randomly sample fixed number of gut matches
        chosen = np.random.choice(num_close_macth, num_close_macth, replace = num_close_macth < num_match)
        chosen = torch.from_numpy(chosen).long().to(xyz_batch.device)
        
        result_p[i, ...] = feat_p[:, chosen]
        result_v[i, ...] = feat_v[:, idx[chosen]]
    
    return result_p, result_v
        

if __name__ == "__main__":
    pass
    