# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F


from .unet_parts import *


class UNet3d(nn.Module):
    def __init__(self, n_channels, n_classes, n_hidden, track_running=True):
        super(UNet3d, self).__init__()
        self.inc = inconv(n_channels, n_hidden, track_running=track_running)
        self.down1 = down(n_hidden, n_hidden*2, track_running=track_running)
        self.down2 = down(n_hidden*2, n_hidden*4, track_running=track_running)
        self.down3 = down(n_hidden*4, n_hidden*8, track_running=track_running)
        self.down4 = down(n_hidden*8, n_hidden*16, track_running=track_running)
        self.down5 = lowest_down2(n_hidden*16, n_hidden*16, track_running=track_running)
        self.up1 = up(n_hidden*32, n_hidden*8, track_running=track_running)
        self.up2 = up(n_hidden*16, n_hidden*4, track_running=track_running)
        self.up3 = up(n_hidden*8, n_hidden*2, track_running=track_running)
        self.up4 = up(n_hidden*4, n_hidden, track_running=track_running)
        self.up5 = up(n_hidden*2, n_hidden, track_running=track_running)
        self.outc = outconv(n_hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return(x)
