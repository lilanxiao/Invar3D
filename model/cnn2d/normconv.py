"""
normalize the depth in local sliding window so that the conv net is invariant to absolute depth
Same as the RDConv in https://www.bmvc2021-virtualconference.com/assets/papers/0501.pdf
"""


# https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: slow! perhaps not optimized by cuDNN
class NormDepthConv2d(nn.Module):
    def __init__(self, padding, kernel_size, stride=1, input_dim=1, output_dim=64, offset=0.01) -> None:
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim, kernel_size, kernel_size))
        self.offset = offset
        torch.nn.init.kaiming_normal_(self.weight.data)
    
    def forward(self, x:torch.Tensor):
        # x: (B, 1, H, W)
        B, _, H, W = x.size()
        x_uf = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)      # (B, K*K, H*W)
        zero_mask = (x_uf > 1e-4).float()                                                   # always keep zeros
        # x_ref = torch.max(x_uf, dim=1, keepdim=True)[0]
        x_ref = torch.sum(x_uf, dim=1, keepdim=True) / (torch.sum(zero_mask, dim=1, keepdim=True) + 1e-6)
        # x_ref = x_uf[:, 4:5, :]
        x_uf_normalized = (x_ref - x_uf) * zero_mask                                        # (B, K*K, H*W)
        # zero_mask_center = (x_ref > 1e-4).float()  
        # x_uf_normalized = (x_ref - x_uf) * zero_mask * zero_mask_center           
        # x_uf_normalized /= self.offset      # scale the input
        out_uf = x_uf_normalized.transpose(1, 2).matmul(self.weight.view(self.weight.size(1), -1).t()).transpose(1, 2)
        out = out_uf.view(B, -1, H//self.stride, W//self.stride)
        return out.contiguous()     # faster with contiguous
