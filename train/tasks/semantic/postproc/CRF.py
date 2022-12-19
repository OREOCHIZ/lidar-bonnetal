#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger


class LocallyConnectedXYZLayer_modifyV2(nn.Module):
    def __init__(self, h, w, sigma, nclasses):
        super().__init__()
        # size of window
        self.h = h
        self.padh = h // 2
        self.w = w
        self.padw = w // 2
        assert (self.h % 2 == 1 and self.w % 2 == 1)  # window must be odd
        self.sigma = sigma
        self.gauss_den = 2 * self.sigma ** 2
        self.nclasses = nclasses
        # self.img_kernel = nn.Conv2d(1, 1, 1, padding=(self.padh, self.padw), padding_mode='circular', bias=False)
        # self.img_kernel.weight = nn.Parameter(torch.tensor([[[[1.]]]]), requires_grad=False)

    def forward(self, range, xyz, prob, mask):
        # softmax size
        N, C, H, W = prob.shape

        # make sofmax zero everywhere input is invalid
        prob = prob * mask.unsqueeze(1).float()

        # get x,y,z for distance (shape N,1,H,W)
        x = xyz[:, 0].unsqueeze(1)
        y = xyz[:, 1].unsqueeze(1)
        z = xyz[:, 2].unsqueeze(1)
        r = range.unsqueeze(1)
        new_r = self.img_kernel(r)

        new_x = self.img_kernel(x)
        new_y = self.img_kernel(y)
        new_z = self.img_kernel(z)

        # im2col in size of window of input (x,y,z separately)
        window_x = F.unfold(new_x, kernel_size=(self.h, self.w))
        # 2D PV 로 표현된 x value 를 3 * 5 kernel size 로 나눔

        center_x = F.unfold(x, kernel_size=(1, 1),
                            padding=(0, 0))  # 해당 커널의 중앙값에 해당함

        window_y = F.unfold(new_y, kernel_size=(self.h, self.w))

        center_y = F.unfold(y, kernel_size=(1, 1),
                            padding=(0, 0))

        window_z = F.unfold(new_z, kernel_size=(self.h, self.w))

        center_z = F.unfold(z, kernel_size=(1, 1),
                            padding=(0, 0))

        window_r = F.unfold(new_r, kernel_size=(self.h, self.w))
        center_r = F.unfold(new_r, kernel_size=(1, 1), padding=(0, 0))

        # sq distance to center (center distance is zero)
        unravel_dist2 = (window_r - center_r) ** 2  # 거리값을 가진 matrix 를 하나 만들어주고

        # weight input distance by gaussian weights
        unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)  # exp (- (pi- pj)^2 / 2*sigma^2)
        # 위와 같이 가공하는 방식이 뭐냐면, mean = center, var = gauss_den
        # 로 하는 어떤 distribution 이 존재할 때 ~ N(center, gauss_den) 그에 대해 주변 3 * 5 의 포인트가 정규 분포를 따를 수 있도록 함.
        # 평균은 0이고, 분산은 1인 값인 정규분포가 될 수 있도록 ...
        # 또한 위의 것은 확률 분포가 되기 때문에, 중심에서 멀어진 점이면 unravel_gaussian 값이 작을 것이고 (그니까, 멀리 떨어져 있어서)
        # 중심과 가까운 점이라면 unravel_gaussian 값이 큰 값이 될 것이다.
        # 그리고 이 unravel_gaussian 는 확률 분포이기 때문에, 0 ~ 1 사이의 값을 가질 수 있게 된다.
        # 거기에 본래의 soft max 값을 곱해주는 셈이니 중심픽셀에서 멀리 떨어진 점일수록 softmax 가 작아지고 가까운 점일 수록 softmax 가 커질 것.

        # im2col in size of window of softmax to reweight by gaussian weights from input
        cloned_softmax = prob.clone()
        for i in range(self.nclasses):
            # get the softmax for this class
            c_softmax = prob[:, i].unsqueeze(1)  # c_softmax = 4 * 1 * 64 * 512
            # unfold this class to weigh it by the proper gaussian weights
            unravel_softmax = F.unfold(c_softmax,
                                       kernel_size=(self.h, self.w),
                                       padding=(self.padh, self.padw))  # unravel_softmax == 4 * 15 * (32768)
            unravel_w_softmax = unravel_softmax * unravel_gaussian  # 4 * 15 * 32768 , Compatibility Transformation?
            # 이렇게 곱하면 결국
            # add dimenssion 1 to obtain the new softmax for this class
            unravel_added_softmax = unravel_w_softmax.sum(dim=1).unsqueeze(
                1)  # 거리에 대해서 위에서 구한 값들을 다 더하는 과정이라고 생각해주면 용이하다... 3*5 내부에 들어와 있는 값을.
            # fold it and put it in new tensor
            added_softmax = unravel_added_softmax.view(N, H, W)
            cloned_softmax[:, i] = added_softmax

        return cloned_softmax  # ([4, 20, 64, 512])


class LocallyConnectedXYZLayer_modify(nn.Module):
    def __init__(self, h, w, sigma, nclasses):
        super(LocallyConnectedXYZLayer_modify, self).__init__()
        # size of window
        self.h = h
        self.padh = h // 2
        self.w = w
        self.padw = w // 2
        assert (self.h % 2 == 1 and self.w % 2 == 1)  # window must be odd
        self.sigma = sigma
        self.gauss_den = 2 * self.sigma ** 2
        self.nclasses = nclasses
        self.img_kernel = nn.Conv2d(1, 1, 1, padding=(self.padh, self.padw), padding_mode='circular', bias=False)
        self.img_kernel.weight = nn.Parameter(torch.tensor([[[[1.]]]]), requires_grad=False)

    def forward(self, xyz, softmax, mask):
        # softmax size
        N, C, H, W = softmax.shape

        # make sofmax zero everywhere input is invalid
        softmax = softmax * mask.unsqueeze(1).float()

        # get x,y,z for distance (shape N,1,H,W)
        x = xyz[:, 0].unsqueeze(1)
        y = xyz[:, 1].unsqueeze(1)
        z = xyz[:, 2].unsqueeze(1)

        new_x = self.img_kernel(x)
        new_y = self.img_kernel(y)
        new_z = self.img_kernel(z)

        # im2col in size of window of input (x,y,z separately)
        window_x = F.unfold(new_x, kernel_size=(self.h, self.w))
        # 2D PV 로 표현된 x value 를 3 * 5 kernel size 로 나눔

        center_x = F.unfold(x, kernel_size=(1, 1),
                            padding=(0, 0))  # 해당 커널의 중앙값에 해당함

        window_y = F.unfold(new_y, kernel_size=(self.h, self.w))

        center_y = F.unfold(y, kernel_size=(1, 1),
                            padding=(0, 0))

        window_z = F.unfold(new_z, kernel_size=(self.h, self.w))

        center_z = F.unfold(z, kernel_size=(1, 1),
                            padding=(0, 0))

        # sq distance to center (center distance is zero)
        unravel_dist2 = (window_x - center_x) ** 2 + \
                        (window_y - center_y) ** 2 + \
                        (window_z - center_z) ** 2  # 거리값을 가진 matrix 를 하나 만들어주고

        # weight input distance by gaussian weights
        unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)  # exp (- (pi- pj)^2 / 2*sigma^2)
        # 위와 같이 가공하는 방식이 뭐냐면, mean = center, var = gauss_den
        # 로 하는 어떤 distribution 이 존재할 때 ~ N(center, gauss_den) 그에 대해 주변 3 * 5 의 포인트가 정규 분포를 따를 수 있도록 함.
        # 평균은 0이고, 분산은 1인 값인 정규분포가 될 수 있도록 ...
        # 또한 위의 것은 확률 분포가 되기 때문에, 중심에서 멀어진 점이면 unravel_gaussian 값이 작을 것이고 (그니까, 멀리 떨어져 있어서)
        # 중심과 가까운 점이라면 unravel_gaussian 값이 큰 값이 될 것이다.
        # 그리고 이 unravel_gaussian 는 확률 분포이기 때문에, 0 ~ 1 사이의 값을 가질 수 있게 된다.
        # 거기에 본래의 soft max 값을 곱해주는 셈이니 중심픽셀에서 멀리 떨어진 점일수록 softmax 가 작아지고 가까운 점일 수록 softmax 가 커질 것.

        # im2col in size of window of softmax to reweight by gaussian weights from input
        cloned_softmax = softmax.clone()
        for i in range(self.nclasses):
            # get the softmax for this class
            c_softmax = softmax[:, i].unsqueeze(1)  # c_softmax = 4 * 1 * 64 * 512
            # unfold this class to weigh it by the proper gaussian weights
            unravel_softmax = F.unfold(c_softmax,
                                       kernel_size=(self.h, self.w),
                                       padding=(self.padh, self.padw))  # unravel_softmax == 4 * 15 * (32768)
            unravel_w_softmax = unravel_softmax * unravel_gaussian  # 4 * 15 * 32768 , Compatibility Transformation?
            # 이렇게 곱하면 결국
            # add dimenssion 1 to obtain the new softmax for this class
            unravel_added_softmax = unravel_w_softmax.sum(dim=1).unsqueeze(
                1)  # 거리에 대해서 위에서 구한 값들을 다 더하는 과정이라고 생각해주면 용이하다... 3*5 내부에 들어와 있는 값을.
            # fold it and put it in new tensor
            added_softmax = unravel_added_softmax.view(N, H, W)
            cloned_softmax[:, i] = added_softmax

        return cloned_softmax  # ([4, 20, 64, 512])


class LocallyConnectedXYZLayer(nn.Module):
    def __init__(self, h, w, sigma, nclasses):
        super(LocallyConnectedXYZLayer, self).__init__()
        # size of window
        self.h = h
        self.padh = h // 2
        self.w = w
        self.padw = w // 2
        assert (self.h % 2 == 1 and self.w % 2 == 1)  # window must be odd
        self.sigma = sigma
        self.gauss_den = 2 * self.sigma ** 2
        self.nclasses = nclasses

    def forward(self, xyz, softmax, mask):
        # softmax size
        N, C, H, W = softmax.shape

        # make sofmax zero everywhere input is invalid
        softmax = softmax * mask.unsqueeze(1).float()

        # get x,y,z for distance (shape N,1,H,W)
        x = xyz[:, 0].unsqueeze(1)
        y = xyz[:, 1].unsqueeze(1)
        z = xyz[:, 2].unsqueeze(1)

        # im2col in size of window of input (x,y,z separately)
        window_x = F.unfold(x, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))  # 2D PV 로 표현된 x value 를 3 * 5 kernel size 로 나눔
        center_x = F.unfold(x, kernel_size=(1, 1),
                            padding=(0, 0))  # 해당 커널의 중앙값에 해당함
        window_y = F.unfold(y, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))
        center_y = F.unfold(y, kernel_size=(1, 1),
                            padding=(0, 0))
        window_z = F.unfold(z, kernel_size=(self.h, self.w),
                            padding=(self.padh, self.padw))
        center_z = F.unfold(z, kernel_size=(1, 1),
                            padding=(0, 0))

        # sq distance to center (center distance is zero)
        unravel_dist2 = (window_x - center_x) ** 2 + \
                        (window_y - center_y) ** 2 + \
                        (window_z - center_z) ** 2  # 거리값을 가진 matrix 를 하나 만들어주고

        # weight input distance by gaussian weights
        unravel_gaussian = torch.exp(- unravel_dist2 / self.gauss_den)  # exp (- (pi- pj)^2 / 2*sigma^2)
        # 위와 같이 가공하는 방식이 뭐냐면, mean = center, var = gauss_den
        # 로 하는 어떤 distribution 이 존재할 때 ~ N(center, gauss_den) 그에 대해 주변 3 * 5 의 포인트가 정규 분포를 따를 수 있도록 함.
        # 평균은 0이고, 분산은 1인 값인 정규분포가 될 수 있도록 ...
        # 또한 위의 것은 확률 분포가 되기 때문에, 중심에서 멀어진 점이면 unravel_gaussian 값이 작을 것이고 (그니까, 멀리 떨어져 있어서)
        # 중심과 가까운 점이라면 unravel_gaussian 값이 큰 값이 될 것이다.
        # 그리고 이 unravel_gaussian 는 확률 분포이기 때문에, 0 ~ 1 사이의 값을 가질 수 있게 된다.
        # 거기에 본래의 soft max 값을 곱해주는 셈이니 중심픽셀에서 멀리 떨어진 점일수록 softmax 가 작아지고 가까운 점일 수록 softmax 가 커질 것.

        # im2col in size of window of softmax to reweight by gaussian weights from input
        cloned_softmax = softmax.clone()
        for i in range(self.nclasses):
            # get the softmax for this class
            c_softmax = softmax[:, i].unsqueeze(1)  # c_softmax = 4 * 1 * 64 * 512
            # unfold this class to weigh it by the proper gaussian weights
            unravel_softmax = F.unfold(c_softmax,
                                       kernel_size=(self.h, self.w),
                                       padding=(self.padh, self.padw))  # unravel_softmax == 4 * 15 * (32768)
            unravel_w_softmax = unravel_softmax * unravel_gaussian  # 4 * 15 * 32768 , Compatibility Transformation?
            # 이렇게 곱하면 결국
            # add dimenssion 1 to obtain the new softmax for this class
            unravel_added_softmax = unravel_w_softmax.sum(dim=1).unsqueeze(
                1)  # 거리에 대해서 위에서 구한 값들을 다 더하는 과정이라고 생각해주면 용이하다... 3*5 내부에 들어와 있는 값을.
            # fold it and put it in new tensor
            added_softmax = unravel_added_softmax.view(N, H, W)
            cloned_softmax[:, i] = added_softmax

        return cloned_softmax  # ([4, 20, 64, 512])


class CRF(nn.Module):
    def __init__(self, params, nclasses):
        super(CRF, self).__init__()
        self.params = params
        self.iter = nn.Parameter(torch.tensor(params["iter"]),
                                       requires_grad=False)
        # self.iter = self.params["iter"]
        self.lcn_size = nn.Parameter(torch.tensor([params["lcn_size"]["h"],
                                                         params["lcn_size"]["w"]]),
                                           requires_grad=False)
        self.xyz_coef = nn.Parameter(torch.tensor(params["xyz_coef"]),
                                           requires_grad=False).float()
        self.xyz_sigma = nn.Parameter(torch.tensor(params["xyz_sigma"]),
                                            requires_grad=False).float()

        self.nclasses = nclasses
        print("Using CRF!")

        # define layers here
        # compat init
        self.compat_kernel_init = np.reshape(np.ones((self.nclasses, self.nclasses)) - np.identity(self.nclasses),[self.nclasses, self.nclasses, 1, 1])

        # bilateral compatibility matrixes
        self.compat_conv = nn.Conv2d(self.nclasses, self.nclasses, 1)
        self.compat_conv.weight = nn.Parameter(torch.from_numpy( self.compat_kernel_init).float() * self.xyz_coef, requires_grad=True)

        # locally connected layer for message passing
        # self.local_conn_xyz = LocallyConnectedXYZLayer_modifyV2(self.params["lcn_size"]["h"],\
        #                                                         self.params["lcn_size"]["w"],\
        #                                                         self.params["xyz_coef"],\
        #                                                         self.nclasses)

    def forward(self, input, prob, mask):
        # use xyz
        xyz = input[:, 1:4]  # input = x, y, z, range, intensity
        range = input[:, 0]

        self.local_conn_xyz = LocallyConnectedXYZLayer_modifyV2(self.params["lcn_size"]["h"],\
                                                                self.params["lcn_size"]["w"],\
                                                                self.params["xyz_coef"],\
                                                                self.nclasses)
        # iteratively
        for i in range(3):
            # message passing as locally connected layer
            print("here? ")
            locally_connected = self.local_conn_xyz(range, xyz, prob, mask)
            # input 을 x,y,z 말고 range 를 사용하고

            # reweigh with the 1x1 convolution
            reweight_softmax = self.compat_conv(locally_connected)
            # self.compat_conv = nn.Conv2d(self.nclasses, self.nclasses, 1) 20, 20, 1
            # 커널사이즈 1*1의 20(input feature 수)* 20 (output feature 수)
            # 국부적으로 가까운지의 확률 분포값까지 곱해준 거에 1*1 conv 를 먹여서, soft max 값을 reweight 하는 과정이라고 한다. local embedding feature 라고 해보자

            # add the new values to the original softmax
            reweight_softmax = reweight_softmax + prob  # 기존 softmax 에 넣어서

            # lastly, renormalize
            prob = F.softmax(reweight_softmax, dim=1)  # 0 ~ 1 사이로 노말라이즈 해줌

        return prob


