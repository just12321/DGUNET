import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModel
from model.modules import CAB, CBR, CR
from einops import rearrange


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels=None):
        out_channels = out_channels if out_channels is not None else in_channels
        super().__init__(
            nn.PixelUnshuffle(2),
            CR(in_channels * 4, out_channels, 3, 1, 1),
            CAB(out_channels, out_channels, 3, 1, 1),
        )


class DWNet(BaseModel):
    """
    Dynamic Weighted Network (DWNet) for image segmentation.
    """

    def __init__(self, in_channels=3, n_classes=1, depth=4, mid_channels=32, ratio = 0.8, min_kernel_size=(12, 12), **kwargs):
        super(DWNet, self).__init__(**kwargs)
        self.proj = CBR(in_channels, mid_channels, 3, 1, 1)
        self.downs = nn.Sequential(
            CBR(3, mid_channels, 3, 1, 1), 
            *[Down(mid_channels) for _ in range(depth)],
            nn.Conv2d(mid_channels, mid_channels * n_classes, 3, 1, 1)
        )
        self.ratio = ratio
        self.min_kernel_size = min_kernel_size
        self.n_classes = n_classes

    def forward(self, x):
        weight = self.downs(x)
        b, c, h, w = weight.shape
        weight = rearrange(weight, 'b (c n) h w -> (b n) c h w', n=self.n_classes) / (c ** 0.5)
        inc = rearrange(self.proj(x), 'b c h w -> () (b c) h w')
        pl = (h - 1) >> 1
        pr = h - 1 - pl
        pt = (w - 1) >> 1
        pb = w - 1 - pt
        inc = F.pad(inc, (pt, pb, pl, pr))
        y = F.conv2d(inc, weight, groups=b)
        y = rearrange(y, '() (b n) h w -> b n h w', n=self.n_classes)
        self.pre = {
            'mask': y
        }
        return y
