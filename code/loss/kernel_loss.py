import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in
             range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in
                                        range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.ones(1).to(kernel.device), torch.sum(kernel))


class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size, scale_factor=.5):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2
        # wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(),
                               requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        b, _, _, _ = kernel.shape
        losses = []
        for _b in range(b):
            kernel_map = torch.squeeze(kernel[_b:_b + 1, :, :, :])
            r_sum, c_sum = torch.sum(kernel_map, dim=1).reshape(1, -1), torch.sum(kernel_map, dim=0).reshape(1, -1)
            loss_cur = self.loss(torch.squeeze(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel_map),
                                                            torch.matmul(c_sum, self.indices) / torch.sum(
                                                                kernel_map)))), self.center)
            losses.append(loss_cur)
        loss_sum = sum(losses) / b

        return loss_sum


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, torch.zeros_like(kernel))


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """

    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.5
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        kernel = torch.abs(kernel)

        power_kernel = torch.zeros_like(kernel)
        mask = kernel > 0
        power_kernel[mask] = kernel[mask] ** self.power

        return self.loss(power_kernel, torch.zeros_like(kernel))
        # return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
