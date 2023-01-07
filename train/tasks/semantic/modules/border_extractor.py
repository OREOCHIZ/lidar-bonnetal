#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
# from laserscan_modifiied import *


class borderExtracter(nn.Module):

    def __init__(self, nclasses, ignore_class):
        super().__init__()
        self.nclasses = nclasses
        self.include_idx = list(range(nclasses))
        self.exclude_idx = self.include_idx.pop(ignore_class)
        self.erode_kernel = self.make_erode_kernel(self.nclasses)

        # self.one_hot_label =

    def make_erode_kernel(self, nclasses):
        erode_kernel = torch.zeros((nclasses, 1, 3, 3)).cuda()
        erode_kernel[:] = torch.tensor([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])

        return erode_kernel

    def make_onehot_label(self, nclasses, label, include_idx, exclude_idx):
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        one_hot_label = F.one_hot(label.long(), num_classes=nclasses)

        one_hot_label = one_hot_label.permute(2, 0, 1)
        one_hot_label = one_hot_label.view(1, one_hot_label.shape[0], one_hot_label.shape[1], one_hot_label.shape[2])

        one_hot_label[:, include_idx] = one_hot_label[:, include_idx] + one_hot_label[:, exclude_idx]

        return one_hot_label.float()

    def get_border_from_label(self, label, erode_iter=1):
        erode_input = self.make_onehot_label(self.nclasses, label, self.include_idx, self.exclude_idx)

        kernel_sum = self.erode_kernel[0][0].sum()

        for _ in range(erode_iter):
            eroded_output = F.conv2d(erode_input, self.erode_kernel, groups=self.nclasses, padding=1)
            eroded_output = (eroded_output == kernel_sum).float()
            erode_input = eroded_output

        background_mask = eroded_output[:, self.exclude_idx] == 1

        eroded_bodies = (eroded_output.sum(1, keepdim=True) == 1)
        eroded_bodies = eroded_bodies + background_mask

        borders = 1 - eroded_bodies.float()

        return borders[0, 0, :, :]

    def visualize_border_plt(self, borders):
        plt.matshow(borders[0, 0, :, :])


if __name__ == "__main__":
    label_name = "/media/se0yeon00/Samsung_T5/SEMANTIC_KITTI/kitti/dataset/sequences/08/labels/000651.label"
    bin_name = "/media/se0yeon00/Samsung_T5/SEMANTIC_KITTI/kitti/dataset/sequences/08/velodyne/000651.bin"

    import yaml

    DATA = yaml.safe_load(open("semantic-kitti.yaml", 'r'))

    test = SemLaserScan(project=True, H=64, W=2048, sem_color_dict=DATA["color_map"])
    test.colorize()
    test.open_scan(bin_name, act_aug=False, rotate_aug=True, flip_aug=True, scale_aug=True, transform=True)
    test.open_label(label_name)

    map_sem_label = test.map(test.proj_sem_label, DATA['learning_map'])
    print(np.unique(map_sem_label, return_counts=True))
    be = borderExtracter(20, 0)
    border_mask = be.get_border_from_label(map_sem_label, erode_iter=1)
    be.visualize_border_plt(border_mask)
