import torch


def gather_normalize_2d3d(fm:torch.Tensor, 
                    fm_mask:torch.Tensor, 
                    seeds_features:torch.Tensor, 
                    ind2d:torch.Tensor, 
                    ind3d:torch.Tensor):
    B, C ,_ ,_ = fm.size()
    fm_lin = fm.view(B, C, -1)
    fm_lin_sample = _gather(fm_lin, fm_mask)    # (B, C, num_samples)
    f2d = _gather(fm_lin_sample, ind2d)         # (B, C, num_match)
    f3d = _gather(seeds_features, ind3d)        # (B, C, num_match)
    f2d = torch.nn.functional.normalize(f2d, dim=1)
    f3d = torch.nn.functional.normalize(f3d, dim=1)
    return f2d, f3d


def contrast_in_bacth(f1:torch.Tensor, f2:torch.Tensor, t:float, symmetric=True):
    """use all other points in the batch as negative samples"""
    B, C, N = f1.size()
    # reshape features 
    f1 = f1.transpose(0, 1).contiguous().view(1, C, B*N) # (B,C,N) -> (C,B,N) -> (1,C,B*N)
    f2 = f2.transpose(0, 1).contiguous().view(1, C, B*N)
    
    logit1 = torch.bmm(f1.transpose(1, 2), f2)          # (1, B*N, B*N)
    label = torch.arange(N*B).long().unsqueeze(0).to(logit1.device)   # (1, B*N)
    loss = torch.nn.CrossEntropyLoss()(logit1/t, label)
    pred = torch.argmax(logit1, dim=1)                              # (B, num_match)    
    acc1 = (pred == label)
    acc1 = torch.sum(acc1.float())/( acc1.size(0) * acc1.size(1))    
    
    if symmetric:
        logit2 = torch.bmm(f2.transpose(1, 2), f1)          # (1, B*N, B*N)
        loss += torch.nn.CrossEntropyLoss()(logit2/t, label)
        loss /= 2.
        pred = torch.argmax(logit2, dim=1)                              # (B, num_match)    
        acc2 = (pred == label)
        acc2 = torch.sum(acc2.float())/( acc2.size(0) * acc2.size(1))
        acc = (acc1 + acc2)/2.
    else:
        acc = acc1
     
    ret = {}
    ret["acc"] = acc
    ret["loss"] = loss
    ret["preds"] = pred
    return ret
        

def contrast_in_scene(f1:torch.Tensor, f2:torch.Tensor, t:float, symmetric=True):
    """only use points in the same scene as negative samples"""
    B, C, N = f1.size()
    logit1 = torch.bmm(f1.transpose(1, 2), f2)          # (B, N, N)
    label = torch.arange(N).long().unsqueeze(0).to(logit1.device)   # (1, N)
    label = label.repeat(B, 1)
    loss = torch.nn.CrossEntropyLoss()(logit1/t, label)
    pred = torch.argmax(logit1, dim=1)                              # (B, num_match)    
    acc1 = (pred == label)
    acc1 = torch.sum(acc1.float())/( acc1.size(0) * acc1.size(1))    
    
    if symmetric:
        logit2 = torch.bmm(f2.transpose(1, 2), f1)          # (B, N, N)
        loss += torch.nn.CrossEntropyLoss()(logit2/t, label)
        loss /= 2.
        pred = torch.argmax(logit2, dim=1)                              # (B, num_match)    
        acc2 = (pred == label)
        acc2 = torch.sum(acc2.float())/( acc2.size(0) * acc2.size(1))
        acc = (acc1 + acc2)/2.
    else:
        acc = acc1

    ret = {}
    ret["acc"] = acc
    ret["loss"] = loss
    ret["preds"] = pred
    return ret


def point_info_nce_loss_2d3d(fm:torch.Tensor, 
                            fm_mask:torch.Tensor, 
                            seeds_features:torch.Tensor, 
                            ind2d:torch.Tensor, 
                            ind3d:torch.Tensor,
                            t:float,
                            symmetric=True,
                            in_batch=False):
    """constrastive loss for depth and point based network"""
    f2d, f3d = gather_normalize_2d3d(fm, fm_mask, seeds_features, ind2d, ind3d)
    
    if in_batch:
        ret = contrast_in_bacth(f2d, f3d, t, symmetric)
    else:
        ret = contrast_in_scene(f2d, f3d, t, symmetric)
    return ret


def point_info_nce_loss(f1, f2, t, 
                        symmetric=True,
                        in_batch=False):
    if in_batch:
        ret = contrast_in_bacth(f1, f2, t, symmetric)
    else:
        ret = contrast_in_scene(f1, f2, t, symmetric)
    return ret


def l1_loss(fm:torch.Tensor, 
            fm_mask:torch.Tensor, 
            seeds_features:torch.Tensor, 
            ind2d:torch.Tensor, 
            ind3d:torch.Tensor):
    B, C ,_ ,_ = fm.size()
    fm_lin = fm.view(B, C, -1)
    fm_lin_sample = _gather(fm_lin, fm_mask)    # (B, C, num_samples)
    f2d = _gather(fm_lin_sample, ind2d)     # B, C, num_match
    f3d = _gather(seeds_features, ind3d)    # B, C, num_match

    loss = torch.nn.functional.smooth_l1_loss(f2d, f3d, reduction="mean")
    ret = {}
    ret["acc"] = torch.Tensor([0])
    ret["loss"] = loss * 10
    return ret


def _gather(feats:torch.Tensor, ind:torch.Tensor) -> torch.Tensor:
    """expand index gather

    Args:
        feats (torch.Tensor): (B, C, N)
        ind (torch.Tensor): (B, M)

    Returns:
        torch.Tensor: (B, C, M)
    """
    C = feats.size(1)
    ind_ext = ind.unsqueeze(1).repeat(1,C,1)
    return torch.gather(feats, 2, ind_ext)
