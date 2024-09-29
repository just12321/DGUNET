"""
Using InstanceNorm
"""
from turtle import forward
from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet import DoubleConv, Down, Up, OutConv, UNet
# from model.coarseSeg import Dilate, Erode, OutlierFilter, UpSample

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, output_padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return self.conv(x1)

class BottleNeck(nn.Module):
    def __init__(self, z=None):
        super().__init__()
        self.proj1 = DoubleConv(1024, 512)
        self.proj2 = DoubleConv(513, 1024)
        self.z = z
    
    def forward(self, x):
        z = self.z if self.z is not None else torch.randn((x.size(0),1,*x.shape[-2:]))
        return self.proj2(torch.cat([self.proj1(x), z.to(x.device)], dim=1))
    
class UNet_C(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, return_feature=False, bottle=nn.Identity()):
        super(UNet_C, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, with_bn=False))
        self.down1 = (Down(64, 128, with_bn=False))
        self.down2 = (Down(128, 256, with_bn=False))
        self.down3 = (Down(256, 512, with_bn=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, with_bn=False))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.return_feature = return_feature
        self.bottle = bottle

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottle(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits if not self.return_feature else (logits, x5)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.pre = nn.Sequential(
            # Erode(),
            # Dilate(),
            # OutlierFilter()
        )
        self.unet = UNet(1, 3, bottle=BottleNeck())

    def forward(self, input, z=None):
        if z is not None:
            self.unet.bottle.z = z
        pre = self.pre(input)
        rand = torch.randn_like(pre)
        return torch.sigmoid(self.unet(rand*pre+(1-rand)*input))
    
class MaskGenerator(nn.Module):
    def __init__(self, out_channels=1):
        super(MaskGenerator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(1, 1024, kernel_size=3, stride=2, output_padding=1),
            DoubleConv(1024, 512, activate=nn.LeakyReLU(0.2, inplace=True)),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, output_padding=1),
            DoubleConv(512, 256, activate=nn.LeakyReLU(0.2, inplace=True)),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, output_padding=1),
            DoubleConv(256, 128, activate=nn.LeakyReLU(0.2, inplace=True)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, output_padding=1),
            DoubleConv(128, 64, activate=nn.LeakyReLU(0.2, inplace=True)),
        )
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        h = self.gen(z)
        h = F.interpolate(h, size=(256, 256), mode='bilinear')
        return self.outc(h)

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x):
        return self.trans(x)

class MaskGan(nn.Module):
    def __init__(self):
        super(MaskGan, self).__init__()
        self.trans = Trans()
        self.gen = MaskGenerator()
        self.cri = MaskCritic()
        self.pre = None

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.size(0), 1, 16, 16).to(x.device)
        latent = self.trans(z)
        mask = self.gen(latent)
        pre = {
            'mask': mask,
            'latent': latent
        }
        self.pre = pre
        return mask
    
    def backward(self, x, optimer, closure:Callable[[Dict], Dict]=None):
        if self.pre is None:
            return None
        if closure is not None:
            pre = self.pre
            self.pre = None
            return closure(pre)
        else:
            loss = {}
            fake = self.pre['mask']
            latent = self.pre['latent']

            optimer['gen'].zero_grad()
            cri_fake, feature_fake = self.cri(fake)
            loss_g = -cri_fake.mean()
            loss_g.backward()
            loss['gen'] = loss_g.item()
            optimer['gen'].step()

            optimer['cri'].zero_grad()
            cri_fake, feature_fake = self.cri(fake.detach())
            cri_real, feature_real = self.cri(x.float())
            loss_d = cri_fake.mean() - cri_real.mean()
            loss['cri'] = loss_d.item()
            loss_gp = gradient_penalty(x, fake.detach(), lambda x:self.cri(x)[0], 10)
            loss_tmp = loss_d + loss_gp
            loss_tmp.backward()
            loss['gp'] = loss_gp.item()
            optimer['cri'].step()
            self.pre = None
            # feature_mean = F.interpolate(feature_real.mean(dim=1, keepdim=True), size=latent.shape[-2:], mode='bilinear')
            # loss_mean = (latent.mean() - feature_mean.mean()) ** 2
            # loss['mean'] = loss_mean.item()
            # loss_std = (latent.std() - feature_mean.std()) ** 2
            # loss['std'] = loss_std.item()
            return loss

class MaskCritic(nn.Module):
    def __init__(self, in_channels=1):
        super(MaskCritic, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            DoubleConv(64, 64, activate=nn.LeakyReLU()),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            DoubleConv(128, 128, activate=nn.LeakyReLU()),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            DoubleConv(256, 256, activate=nn.LeakyReLU()),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            DoubleConv(512, 512, activate=nn.LeakyReLU()),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            DoubleConv(1024, 1024, with_bn=False, activate=nn.LeakyReLU()),
        )
        self.cri = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        h = self.extractor(x)
        return self.cri(h), h

class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()
        self.extractor = UNet_C(in_channels, 1, return_feature=True)
        self.disc = nn.Sequential(
            DoubleConv(1024, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    
    def forward(self, input):
        mask, feature = self.extractor(input)
        return torch.sigmoid(mask), self.disc(feature)
    
class GAG(nn.Module):
    def __init__(self):
        super(GAG, self).__init__()
        self.gen = MaskGenerator()
        self.cri = MaskCritic()
    
    def forawrd(self, x, z=None):
        if z is None:
            z = torch.randn(x.size(0), 1, 16, 16).to(x.device)
        mask = self.gen(z)
        return mask
    
    def judge(self, x):
        return self.cri(x)
    
class MaskGAG(nn.Module):
    def __init__(self):
        super(MaskGAG, self).__init__()
        self.gan1 = GAG()
        self.gan2 = GAG()
        self.pre = None
    
    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.size(0), 1, 16, 16).to(x.device)
            
        mask1 = self.gan1.forawrd(x, z)
        mask2 = self.gan2.forawrd(x, z)
        self.pre = (mask1, mask2)
        return torch.cat([mask1, mask2], dim=2)
    
    def backward(self, mask, optimer, closure:Callable[[Dict], Dict]=None):
        if self.pre is None:
            return None
        if closure is not None:
            pre = self.pre
            self.pre = None
            return closure(pre)
        else:
            loss = {}
            mask1, mask2 = self.pre

            optimer['disc1'].zero_grad()
            disc1, _ = self.gan1.judge(mask.float())
            disc2, _ = self.gan1.judge(mask1.detach())
            disc3, _ = self.gan1.judge(mask2.detach())
            loss_d1 = disc2.mean() + disc3.mean() - disc1.mean()
            loss_gp1 = gradient_penalty(mask.float(), mask1.detach(), lambda x:self.gan1.judge(x)[0], 10)
            loss_tmp1 = loss_d1 + loss_gp1
            loss_tmp1.backward()
            loss['disc1'] = loss_tmp1.item()
            optimer['disc1'].step()

            optimer['disc2'].zero_grad()
            disc1, _ = self.gan2.judge(mask1.detach())
            disc2, _ = self.gan2.judge(mask.float())
            disc3, _ = self.gan2.judge(mask2.detach())
            loss_d2 = disc1.mean() + disc3.mean() - disc2.mean()
            loss_gp2 = gradient_penalty(mask.float(), mask2.detach(), lambda x:self.gan2.judge(x)[0], 10)
            loss_tmp2 = loss_d2 + loss_gp2
            loss_tmp2.backward()
            loss['disc2'] = loss_d2.item() + loss_gp2.item()
            optimer['disc2'].step()
            
            optimer['gen1'].zero_grad()
            disc1, _ = self.gan1.judge(mask1)
            disc2, _ = self.gan2.judge(mask1)
            loss_g1 = -disc1.mean() - disc2.mean()
            loss_g1.backward()
            loss['gen1'] = loss_g1.item()
            optimer['gen1'].step()

            optimer['gen2'].zero_grad()
            disc1, _ = self.gan1.judge(mask2)
            disc2, _ = self.gan2.judge(mask2)
            loss_g2 = -disc1.mean() - disc2.mean()
            loss_g2.backward()
            loss['gen2'] = loss_g2.item()
            optimer['gen2'].step()
            self.pre = None
            return loss

def clip_gradient(model, clip_value):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.data.clamp_(-clip_value, clip_value)

def gradient_penalty(gt, fake, enclosure, lambda_term=10):
    p = torch.rand(gt.size(0), 1, 1, 1).to(gt.device)
    sample = p*gt + (1-p)*fake.detach()
    sample.requires_grad_(True)
    cri = enclosure(sample)
    grad = torch.autograd.grad(cri, sample, grad_outputs=torch.ones_like(cri), create_graph=True, retain_graph=True)[0]
    grad_norm = grad.view(gt.size(0), -1).norm(2, dim=1)
    return ((grad_norm-1)**2).mean()*lambda_term
