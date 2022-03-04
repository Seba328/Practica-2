import torch
import numpy as np
from torch import nn


def generate_conv_layer(input_layer_par: tuple, pool_size=None, ave_pool=True):
    input_num, output_num, kernel_size, stride = input_layer_par
    conv = nn.Conv2d(input_num, output_num, kernel_size, stride, padding=1)
    if pool_size is not None:
        pool = nn.AvgPool2d(pool_size) if ave_pool else nn.MaxPool2d(pool_size)
        return nn.Sequential(conv, nn.ReLU(), pool)
    else:
        return nn.Sequential(conv, nn.ReLU())


class CnnExtractor(nn.Module):
    def __init__(self, img_size: tuple):
        super().__init__()
        self.conv1 = generate_conv_layer(input_layer_par=(img_size[0], 32, (8, 8), (4, 4)))
        self.conv2 = generate_conv_layer(input_layer_par=(32, 64, (4, 4), (2, 2)))
        self.conv3 = generate_conv_layer(input_layer_par=(64, 64, (3, 3), (1, 1)))
        self.conv_set = (self.conv1, self.conv2, self.conv3)
        self.flatten_size = self._cal_size((1, *img_size))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _cal_size(self, img_size):

        x = torch.zeros(img_size)
        for conv in self.conv_set:
            x = conv(x)
        return np.prod(x.shape[1:])

    def conv(self, x: torch.Tensor) -> torch.Tensor:

        for conv in self.conv_set:
            x = conv(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        return x
