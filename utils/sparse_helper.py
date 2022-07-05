import torch
import numpy as np
import MinkowskiEngine as ME
from numpy import ndarray
from torch import Tensor
from typing import List, Dict, Iterable


def collate_dense_sparse(sparse_key:Iterable[str], pairs : int = 1):
    """only stack dense input, keep sparse input as list

    Args:
        sparse_key (Iterable[str]): key of sparse input. The first is for
            sparse coordinate. The second is for sparse features
        pairs (int): if pairs > 1, add surfix to sparse key

    Returns:
        function for Dataloader
    """
    def func(data_list :List[Dict[str, ndarray]]):
        keys = data_list[0].keys()
        ret = {key:[] for key in keys}
        for data in data_list:
            for key in keys:
                ret[key].append(torch.from_numpy(data[key]))

        if pairs == 1:
            # stack dense tensor
            for key in keys:
                if sparse_key[0] is not key and sparse_key[1] is not key:
                    ret[key] = torch.stack(ret[key], axis=0)
            # collate sparse tensor
            coord, feat = ME.utils.sparse_collate(ret[sparse_key[0]], ret[sparse_key[1]])
            ret[sparse_key[0]] = coord
            ret[sparse_key[1]] = feat
        else:
            # stack dense tensor
            for key in keys:
                if sparse_key[0] not in key and sparse_key[1] not in key:
                    ret[key] = torch.stack(ret[key], axis=0)
            # collate sparse tensor
            for i in range(1, pairs+1):
                coord, feat = ME.utils.sparse_collate(ret[sparse_key[0]+str(i)], ret[sparse_key[1]+str(i)])
                ret[sparse_key[0]+str(i)] = coord
                ret[sparse_key[1]+str(i)] = feat
        return ret
    
    return func


def to_device(dic:Dict, device:str):
    """send tensor to device"""
    for key in dic:
        dic[key] = dic[key].to(device)
    return dic


def align_sparse_features(sparse_out:ME.SparseTensor, inds):
    bs, num_match = inds.size()
    sparse_feature = sparse_out.features
    sparse_coord = sparse_out.coordinates
    dims = sparse_feature.size(-1)
    feats = torch.zeros(bs, num_match, dims).to(sparse_out.device)
    for i in range(bs):
        feats[i, ...] = sparse_feature[sparse_coord[:, 0]==i, :][inds[i, :], :]
    feats = feats.transpose(1, 2).contiguous()
    return feats