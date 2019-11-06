# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, track_running=True):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, track_running=True):
        self.track_running_stats=track_running
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25), #0.1
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25), #0.1
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, track_running=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1),
            double_conv(in_ch, out_ch, track_running)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return(x)


class lowest_down2(nn.Module):
    def __init__(self, in_ch, out_ch, track_running=True):
        super(lowest_down2, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1),
            double_conv(in_ch, out_ch, track_running)
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch*2, 3, padding=1),
            nn.BatchNorm3d(out_ch*2, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch*2, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        x = self.conv(x)
        return(x)

class lowest_down(nn.Module):
    def _init__(self, in_ch, out_ch, track_running=True):
        super(lowest_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch*2, 3, padding=1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch*2, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch, track_running_stats=track_running),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.mpconv=nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.mpconv(x)
        x = self.conv(x)
        return(x)



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, track_running=True):
        super(up, self).__init__()

        self.conv = double_conv(in_ch, out_ch, track_running)

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1,scale_factor=2)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return(x)

class leaky_hardtanh(nn.Module):
    def __init__(self, min=-1, max=1, slope=0.01):
        super(leaky_hardtanh, self).__init__()
        self.min = min
        self.max = max
        self.slope = slope

    def forward(self, x):
        x = torch.where(x<self.min, self.min+x*self.slope, x)
        x = torch.where(x>self.max, self.max+x*self.slope, x)
        return(x)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.Hardtanh()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1)
        )

    def forward(self, x):
        x1 = self.conv(x)
        sigma = self.conv2(x)
        out = torch.cat((x1, sigma), dim=1)
        return(out)
