import torch
import torch.nn as nn
from model import flow_pwc
from model.kernel import KernelNet
import torch.nn.functional as F
import math


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    return PWC_Recons(n_colors=args.n_colors, n_sequence=args.n_sequence, extra_RBS=args.extra_RBS,
                      recons_RBS=args.recons_RBS, n_feat=args.n_feat, scale=args.scale, ksize=args.ksize,
                      HR_in=args.HR_in, flow_pretrain_fn=flow_pretrain_fn, device=device)


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


class PWC_Recons(nn.Module):

    def __init__(self, n_colors=3, n_sequence=5, extra_RBS=5, recons_RBS=10, n_feat=32, kernel_size=3,
                 scale=4, ksize=13, HR_in=False, flow_pretrain_fn='.', device='cuda'):
        super(PWC_Recons, self).__init__()
        print("Creating PWC-Recons Net")

        self.ksize = ksize
        self.n_sequence = n_sequence
        self.scale = scale
        self.HR_in = HR_in
        self.device = device

        In_conv = [nn.Conv2d(n_colors, n_feat, kernel_size=3, stride=1, padding=1)]

        Extra_feat = []
        Extra_feat.extend([ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                           for _ in range(extra_RBS)])

        Fusion_conv = [nn.Conv2d(n_feat * n_sequence, n_feat, kernel_size=3, stride=1, padding=1)]

        Recons_net = []
        Recons_net.extend([ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                           for _ in range(recons_RBS)])

        Out_conv = [
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_feat, n_colors, kernel_size=3, stride=1, padding=1)
        ]

        Upsample_layers = []
        for _ in range(int(math.log2(scale))):
            Upsample_layers.append(nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1, bias=True))
            Upsample_layers.append(nn.PixelShuffle(2))

        self.in_conv = nn.Sequential(*In_conv)
        self.extra_feat = nn.Sequential(*Extra_feat)
        self.fusion_conv = nn.Sequential(*Fusion_conv)
        self.recons_net = nn.Sequential(*Recons_net)
        self.out_conv = nn.Sequential(*Out_conv)
        self.upsample_layers = nn.Sequential(*Upsample_layers)
        self.flow_net = flow_pwc.Flow_PWC(pretrain_fn=flow_pretrain_fn, device=device)
        self.kernel_net = KernelNet(in_c=n_colors * n_sequence, ksize=self.ksize)

    def forward(self, input_dict):
        x = input_dict['x']
        mode = input_dict['mode']

        if mode == 'train':
            frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
            est_kernel = self.kernel_net(torch.cat(frame_list, dim=1))

            down_frame_list = [self.blur_down(f, est_kernel, scale=self.scale) for f in frame_list]
            down_recons = self.recons_forward(down_frame_list)

            with torch.no_grad():
                recons = self.recons_forward(frame_list)
            input_cycle = self.blur_down(recons.detach(), est_kernel, scale=self.scale)

            mid_loss = None

            return {
                       'input': frame_list[self.n_sequence // 2],
                       'recons_down': down_recons,
                       'input_cycle': input_cycle,
                       'est_kernel': est_kernel
                   }, mid_loss
        elif mode == 'val':
            frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
            est_kernel = self.kernel_net(torch.cat(frame_list, dim=1))

            recons = self.recons_forward(frame_list)
            input_cycle = self.blur_down(recons, est_kernel, scale=self.scale)

            down_frame_list = [self.blur_down(f, est_kernel, scale=self.scale) for f in frame_list]
            down_recons = self.recons_forward(down_frame_list)

            mid_loss = None

            return {
                       'input': frame_list[self.n_sequence // 2],
                       'input_down': down_frame_list[self.n_sequence // 2],
                       'input_cycle': input_cycle,
                       'recons': recons,
                       'recons_down': down_recons,
                       'est_kernel': est_kernel
                   }, mid_loss
        elif mode == 'infer':
            frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
            est_kernel = self.kernel_net(torch.cat(frame_list, dim=1))
            recons = self.recons_forward(frame_list)

            mid_loss = None

            return {
                       'recons': recons,
                       'est_kernel': est_kernel
                   }, mid_loss
        else:
            raise Exception('Not support mode={}!'.format(mode))

    def recons_forward(self, frame_list):
        frame_feat_list = [self.extra_feat(self.in_conv(frame)) for frame in frame_list]

        base = frame_list[self.n_sequence // 2]
        if not self.HR_in:
            base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)

        warped_feat_list = []
        for i in range(self.n_sequence):
            if not i == self.n_sequence // 2:
                flow = self.flow_net(frame_list[self.n_sequence // 2], frame_list[i])
                warped_feat, _ = self.flow_net.warp(frame_feat_list[i], flow)
                warped_feat_list.append(warped_feat)
            else:
                warped_feat_list.append(frame_feat_list[i])

        fusion_feat = self.fusion_conv(torch.cat(warped_feat_list, dim=1))
        recons_feat = self.recons_net(fusion_feat)
        if self.HR_in:
            recons = self.out_conv(recons_feat)
        else:
            recons = self.out_conv(self.upsample_layers(recons_feat))
        recons = recons + base
        return recons

    def conv_func(self, input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    def blur_down(self, x, kernel, scale):
        b, c, h, w = x.size()
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(self.conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        # down
        blurdown = blur[:, :, ::scale, ::scale]

        return blurdown
