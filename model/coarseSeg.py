"""
Using group-unet
"""
from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gan import MaskCritic, gradient_penalty
from utils.losses import contain_loss
from utils.timer import Timer
from model.base import BaseModel
from model.modules import ClickMap, DistanceMap, PointWiseConv2d, Sum, Mean
import math
from model.unet import GUNet_n, GUNet_tiny, UNet, GUNet
from model.modules import ASPP, GroupConv2d
from utils.utils import filter as filter_
from utils.metrics import Accuracy, Dice, ErodeIou, Iou
class Diffusion(nn.Module):
    def __init__(self, in_channels=3, iter_nums=3):
        super().__init__()
        self.iter = iter_nums
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.diffuse = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
        )
        self.fuse1 = nn.Conv2d(2, 1, 1, bias=False)
        self.filter = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.ReLU()
        )
        self.fuse2 = nn.Conv2d(2, 1, 1, bias=False)
        nn.init.constant_(self.fuse1.weight, 0.5)
        nn.init.constant_(self.fuse2.weight, 0.5)
            

    def forward(self, img, mask):
        res = []
        img = self.proj(img)
        for _ in range(self.iter):
            mask = self.fuse1(torch.cat([mask, self.diffuse(img * mask)], dim=1))
            res.append(mask)
        return res if len(res) > 0 else [mask]
        return self.fuse2(torch.cat([mask,
                                      F.interpolate(self.filter(F.adaptive_avg_pool2d(mask, (mask.size(-2)//8, mask.size(-1)//8))), size=mask.shape[-2:])], dim=1))

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, groups=3, stride=8):
        super().__init__()
        step = int(math.log(stride, 2))
        channels = [in_channels*groups] + [groups*64*(2**i) for i in range(step)]#groups*[in_channel, 64,128,256...]
        self.extractor = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, groups=groups, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=3, groups=groups, dilation=3, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1, groups=groups, bias=False),
                    nn.BatchNorm2d(channels[i+1])
                ) for i in range(step)
            ]
        )
    
    def forward(self, x):
        res = []
        for extractor in self.extractor:
            x = extractor(x)
            res.append(x)
        return res

class UpSample(nn.Module):
    def __init__(self, in_channels=256, stride=8, groups = 2):
        super().__init__()
        step = int(math.log(stride, 2))
        channels = [in_channels*2] + [in_channels//(2**i) for i in range(step)]#2*[in_channel, in_channel//2, in_channel//4...]
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=3, stride=2, bias=False, output_padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1]//2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1]//2, channels[i+1]//2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]//2),
                    nn.ReLU(inplace=True),
                ) for i in range(step)
            ]
        )
        self.groups = groups
    def forward(self, x, res):
        for idx, layer in enumerate(self.upsample):
            pre = torch.chunk(res[-1-idx], self.groups, dim=1)[-1]
            x = layer(torch.cat([x, pre], dim=1))
        return x

class Dilate(nn.Module):
    def __init__(self, kernel_size=5, iter_num=1, mode='conv', threshold=0.1):
        super().__init__()
        self.dilate = nn.Conv2d(1, 1, kernel_size, 1, kernel_size//2, bias=False) if mode == 'conv' \
            else nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.iter_num = iter_num
        self.mode = mode
        self.threshold = max(threshold, 0) + 1e-6
        if mode == 'conv':nn.init.constant_(self.dilate.weight, 1)
    def forward(self, x):
        _, _, h, w = x.shape
        for _ in range(self.iter_num):
            x = self.dilate(x)
            if self.mode=='conv':x = torch.tanh(x/self.threshold)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

class Erode(nn.Module):
    def __init__(self, kernel_size=5, iter_num=1, mode='conv', threshold=0.1):
        super().__init__()
        self.erode = nn.Conv2d(1, 1, kernel_size, 1, kernel_size//2, bias=False) if mode == 'conv' \
            else nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.iter_num = iter_num
        self.mode = mode
        self.threshold = nn.Parameter(torch.tensor(threshold))
        if mode == 'conv':nn.init.constant_(self.erode.weight, 1)
    def forward(self, x):
        _, _, h, w = x.shape
        threshold = self.erode(torch.fill_(torch.empty_like(x), self.threshold))
        for _ in range(self.iter_num):
            if self.mode == 'conv':
                x = F.relu(self.erode(x - self.threshold)) - (self.erode(x) - threshold)
                x = torch.tanh(x/self.threshold.detach())
            else:
                x = -self.erode(-x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

class NarrowCompensation(nn.Module):
    """
    For each patch, each value should multiple `h*w/sum`.\n
    Notice that `avg=sum/(h*w)`, hence the following `x/(avg+eps)`\n
    avg: F.interpolate(AvgPool)
    """
    def __init__(self, patch_size=20):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size, padding=patch_size//2)
    def forward(self, mask):
        _, _, h, w = mask.shape
        avgpool = self.pool(mask)
        avg = F.interpolate(avgpool, size=(h, w), mode='bilinear', align_corners=False)
        return mask/(avg+1e-6)

class OutlierFilter(nn.Module):
    """
    Problem: Narrow region could also be filtered. \n
    The empirical analysis shows that it tends to converge on filtering regions whose blank area exceeds the threshold.\n
    * Notice: The results show no significant difference, it means mask itself is a gate, and `torch.sigmoid(self.fuse(...))` can completely replace it 
    """
    def __init__(self, kernel_size=21, threshold=-0.08):
        super().__init__()
        # self.scale1 = nn.Conv2d(1, 1, 1) #adjust to fit the following `0` padding
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.sum = nn.Conv2d(1, 1, kernel_size) #sum
        # self.scale2 = nn.Conv2d(1, 1, 1) # scale it
        # self.scale2 = lambda x: x * -2
        # self.filter = nn.Sequential(
        #     nn.Conv2d(1, 1, 1), 
        #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2), # sum
        #     nn.ReLU(), # filter small areas
        #     nn.Conv2d(1, 1, 1), # scale it
        #     nn.Sigmoid() # to be the mask of a mask 
        # )
        # nn.init.constant_(self.scale1.weight, 1.) # 0.001
        # nn.init.constant_(self.scale1.bias, -0.5)    # 0.5
        nn.init.constant_(self.sum.weight, 0.008)   # 0.015
        nn.init.constant_(self.sum.bias, 2 - threshold) # -0.05
        # nn.init.constant_(self.scale2.weight, 0.5) # 0.5
        # nn.init.constant_(self.scale2.bias, 1.5) # 1.5
        # self.sum = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2)
        # nn.init.constant_(self.sum.weight, 0.001)
        # nn.init.constant_(self.sum.bias, threshold)
        # self.dilate = Dilate()
        # self.erode = Erode()
    
    def forward(self, mask):
        return mask * torch.sigmoid(F.leaky_relu(self.sum(self.pad(mask))))
        # return torch.clamp(self.dilate(self.erode(mask)), 0, 1)
        statistic = self.sum(mask-0.5) 
        mask_mask = torch.sigmoid(F.relu(statistic) * 100)
        return mask * mask_mask

class Constrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            GroupConv2d(4, 64, 1, groups=32),
            nn.LeakyReLU(),
            GroupConv2d(64, 128, 1, groups=64),
            nn.LeakyReLU(),
        )
        self.trans = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1, bias=False),
        )
        
    def forward(self, x, mask):
        embed = self.embed(torch.cat([x, mask], dim=1))
        return self.trans(embed)

class ConstrainV0_0(Constrain):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(4, 64, 1, groups=4),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 1, groups=64),
            nn.LeakyReLU(),
        )

class ConstrainV0_1(Constrain):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(4, 128, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 512, 1),
            nn.LeakyReLU(),
        )

class ConstrainV0_2(Constrain):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            GroupConv2d(4, 128, 1, groups=128),
            nn.LeakyReLU(),
            GroupConv2d(128, 512, 1, groups=512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 1, groups=512, bias=False),
        )

class CoarseSeg(BaseModel):
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__()
        self.clickmap = ClickMap()
        self.distancemap = DistanceMap()
        self.local_diffusion = Diffusion(iter_nums=3)
        self.constrain = Constrain()
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.unet = nn.Sequential(
            GUNet_n(n_channels=4, n_classes=1, bottle=ASPP(512,[6,12,18],512)),
            # NarrowCompensation(patch_size=10),
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            # nn.Sigmoid()
        )
        self.dilate = Dilate()
        self.erode = Erode()
        self.outlier = OutlierFilter(kernel_size=31, threshold=-0.065)
        self.cri = MaskCritic()
        self.fuse.weight.data = torch.tensor([[[[1.6]],[[0.2]]]], requires_grad=True)
        self.pre = None

    def latent(self, mask):
        kernel = torch.ones(1, 1, 21, 21, device=mask.device)
        padded = F.pad(mask*2, [10]*4, mode='constant', value=1)
        value = F.conv2d(padded, kernel)
        zeros = torch.zeros_like(mask)
        return torch.where(value>110, mask, zeros)

    def predict(self, img, points = None, pre_mask = None):
        return self.forward(img, points, pre_mask)['mask']

    def forward(self, img, points = None, pre_mask = None):
        with Timer("prepare", True, self.trace_time):
            b, _, H, W = img.shape
            pre_mask = pre_mask if pre_mask is not None else torch.zeros(b, H, W, device=img.device)
            if points is None:
                C = 2
                clickmap = torch.ones(img.shape[0]*C, 1, H, W, device=img.device)
                distancemap = torch.ones(img.shape[0], H, W, device=img.device)
            else:
                _, C, n, s = points.shape
                clickmap = self.clickmap(img, points.view(-1, n, s))
                distancemap = self.distancemap(clickmap).view(b, C, H, W)
                distancemap = distancemap[:,0]-distancemap[:,1]

            pre_mask = pre_mask + distancemap
            pre_mask = torch.clamp(pre_mask, 0, 1).unsqueeze(1)
        with Timer("constrain", True, self.trace_time):
            local_x = self.constrain(img, pre_mask)
        with Timer("local_diffusion", True, self.trace_time):
            latent_constrain = self.local_diffusion(img, local_x)
        with Timer("unet", True, self.trace_time):
            coarse_mask =  self.unet(torch.cat([img, latent_constrain[-1].detach()], dim=1))
        with Timer("fuse", True, self.trace_time):
            y = torch.sigmoid(self.fuse(torch.cat([coarse_mask , latent_constrain[-1]], dim=1)))
        self.pre =  {
            'img': img,
            'mask': y,
            'local': local_x,
            'latent': latent_constrain,
            'residual': coarse_mask, 
            'coarsemask': torch.sigmoid(coarse_mask),
        }
        return y
    
    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None, clear_stored=True):
        super().backward(x, optimer, closure, clear_stored)
        loss = {}
        mask = self.pre['mask']
        img = self.pre['img']
        latent = self.pre['latent'][-1]

        optimer.zero_grad()
        loss_bce = F.binary_cross_entropy(mask, x.float())
        loss['bce'] = loss_bce.item()
        loss_contain = contain_loss(latent, x.float(), False)
        loss['contain'] = loss_contain.item()
        loss_tmp = loss_bce * 0.8 + loss_contain * 0.2
        loss_tmp.backward()
        optimer.step()

        if clear_stored:self.pre = None
        return loss
    
    def metrics(self, img, mask, points = None, pre_mask = None):
        pred = self.forward(img, points, pre_mask).squeeze().detach().cpu().numpy()
        mask = mask.squeeze().detach().cpu().numpy()
        metrics = {}
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
        filtered = filter_(pred)
        ms = {
            'Acc': Accuracy,
            'Iou': Iou,
            'F1': Dice,
            'E_Iou': ErodeIou,
        }
        for k, v in ms.items():
            metrics[k] = v(pred, mask)
            metrics[f'F_{k}'] = v(filtered, mask)
        return metrics

    def memo(self):
        return """
            Using Grouped-Convolution as Embedding.
        """
    
class CoarseSegv0_0(CoarseSeg):
    """
    Using seperatable convolution to replace Grouped-Convolution
    """
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__(in_channels, stride, head_num)
        self.constrain = ConstrainV0_0()
    
    def memo(self):
        return """
            Using seperatable convolution to replace Grouped-Convolution.
        """

class CoarseSegv0_1(CoarseSeg):
    """
    Using vanilla convolution to replace Grouped-Convolution
    """
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__(in_channels, stride, head_num)
        self.constrain = ConstrainV0_1()
    
    def memo(self):
        return """
            Using vanilla convolution to replace Grouped-Convolution.
        """

class CoarseSegv0_2(CoarseSeg):
    """
    Using Grouped-Convolution
    """
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__(in_channels, stride, head_num)
        self.constrain = ConstrainV0_2()
    
    def memo(self):
        return """
            Using Grouped-Convolution completely grouped on channels.
        """

class CoarseSegv0_3(CoarseSeg):
    """
    Using seperatable convolution and tiny GUet
    """
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__(in_channels, stride, head_num)
        self.constrain = ConstrainV0_0()
        self.unet = GUNet_tiny(n_channels=4, n_classes=1, bottle=ASPP(128,[6,12,18],128), as_layer=True)
    
    def memo(self):
        return """
            Using seperatable convolution and tiny GUet.
        """
 
class CoarseSegWithDisc(nn.Module):
    def __init__(self, in_channels=3, stride=16, head_num=4):
        super().__init__()
        self.seg = CoarseSeg(in_channels=in_channels, stride=stride, head_num=head_num)
        self.cri = MaskCritic(in_channels+1)
        self.pre = None
    
    def forward(self, img, points = None, pre_mask = None):
        pre = self.seg.forward(img, points, pre_mask)
        self.pre = pre
        return pre['mask']

    def getfig(self, img=None):
        if img is None:
            if self.pre is None:
                return None
        else:
            self.forward(img)
        store = self.pre

    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None):
        if self.pre is None:
            return None
        if closure is not None:
            pre = self.pre
            self.pre = None
            closure(pre)
        loss = {}
        mask = self.pre['mask']
        img = self.pre['img']
        latent = self.pre['latent'][-1]

        optimer['cri'].zero_grad()
        cri_gt, _ = self.cri(torch.cat([img,x.float()],dim=1))
        cri_fake, _ = self.cri(torch.cat([img,mask.detach()],dim=1))
        # loss_cri = F.binary_cross_entropy_with_logits(cri_gt, torch.ones_like(cri_gt)) + \
        #             F.binary_cross_entropy_with_logits(cri_fake, torch.zeros_like(cri_fake))
        loss_cri = cri_gt.mean() - cri_fake.mean() + gradient_penalty(x, mask, lambda y: self.cri(torch.cat([img,y], dim=1))[0])
        loss_cri.backward()
        loss['cri'] = loss_cri.item()
        optimer['cri'].step()

        optimer['seg'].zero_grad()
        cri_fake, _ = self.cri(torch.cat([img,mask],dim=1))
        loss_seg = F.binary_cross_entropy_with_logits(cri_fake, torch.zeros_like(cri_fake))
        # loss_seg = cri_fake.mean()
        loss['gen'] = loss_seg.item()
        loss_bce = F.binary_cross_entropy(mask, x.float())
        loss['bce'] = loss_bce.item()
        loss_contain = contain_loss(latent, x.float(), False)
        loss['contain'] = loss_contain.item()
        loss_tmp = loss_seg * 0.01 + loss_bce * 0.8 + loss_contain * 0.1
        # loss_tmp = loss_bce
        loss_tmp.backward()
        optimer['seg'].step()

        self.pre = None
        return loss

    def metrics(self, img, mask, points = None, pre_mask = None):
        return self.seg.metrics(img, mask, points, pre_mask)

