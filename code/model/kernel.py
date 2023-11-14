import torch.nn as nn


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(False), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res


class KernelNet(nn.Module):
    def __init__(self, in_c=3, ksize=13):
        super(KernelNet, self).__init__()
        self.ksize = ksize

        self.in_conv = nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.body1 = nn.Sequential(
            RCAB(n_feat=64),
            RCAB(n_feat=64)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(self.ksize)
        self.body2 = nn.Sequential(
            RCAB(n_feat=64),
            RCAB(n_feat=64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True)
        )
        self.fc_net = nn.Sequential(
            nn.Linear(ksize * ksize, 1000, bias=True),
            nn.Linear(1000, ksize * ksize, bias=True),
            nn.Softmax()
        )

    def forward(self, input_tensor):
        b, c, h, w = input_tensor.size()
        out = self.in_conv(input_tensor)
        out = self.body1(out)
        out = self.global_pool(out)
        out = self.body2(out)
        out = out.view(b, self.ksize * self.ksize)
        out = self.fc_net(out)
        est_kernel = out.view(b, 1, self.ksize, self.ksize)
        return est_kernel
