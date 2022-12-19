from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
# from loss_additional import *

class ContrastCELoss(nn.Module):
    def __init__(self, ARCH, loss_w):
        super(ContrastCELoss, self).__init__()
        self.ARCH = ARCH
        self.loss_weight = self.ARCH['contrastive']['loss_weight']

        self.seg_creterion = nn.NLLLoss(weight=loss_w) # loss existing
        self.contrast_creterion = PixelContrastLoss(self.ARCH) # added loss

    def forward(self, feats, outputs, labels):
        # seg_loss = self.seg_creterion(outputs, labels)
        seg_loss = self.seg_creterion(torch.log(outputs.clamp(min=1e-8)), labels)

        # 221215
        _, predict = torch.max(outputs, 1)
        contrast_loss = self.contrast_creterion(feats, labels, predict)

        if self.ARCH['contrastive']['with_embed'] is True:
            return seg_loss + self.loss_weight * contrast_loss

        return seg_loss + 0 * contrast_loss


class ContrastCEDiceLoss(nn.Module):
    def __init__(self, ARCH, loss_w):
        super(ContrastCELoss, self).__init__()
        self.ARCH = ARCH
        self.loss_weight = self.ARCH['contrastive']['loss_weight']

        self.seg_creterion = nn.NLLLoss(weight=loss_w)  # loss existing
        self.contrast_creterion = PixelContrastLoss(self.ARCH)  # added loss

    def forward(self, feats, outputs, labels):
        seg_loss = self.seg_creterion(torch.log(outputs.clamp(min=1e-8)), labels)
        _, predict = torch.max(outputs, 1)
        contrast_loss = self.contrast_creterion(feats, labels, predict)

        if self.ARCH['contrastive']['with_embed'] is True:
            return seg_loss + self.loss_weight * contrast_loss

        return seg_loss + 0 * contrast_loss

class PixelContrastLoss(nn.Module):
    def __init__(self, ARCH):
        super(PixelContrastLoss, self).__init__()
        self.ARCH = ARCH
        self.max_samples = self.ARCH['contrastive']['max_samples']
        self.max_views = self.ARCH['contrastive']['max_views']
        self.temperature = self.ARCH['contrastive']['temperature']
        self.base_temperature = self.ARCH['contrastive']['base_temperature']
        self.ignore_label = self.ARCH['contrastive']['loss_w']
        self.queue = None
        self.memory_size = 0


        if self.ARCH['contrastive']['with_memory']:
            self.queue = {}
            self.minor_queue = {}
            self.memory_size = self.ARCH['contrastive']['memory_size']
            # self.register_buffer("pixel_queue", torch.randn(num_classes, self.memory_size, dim))


    def forward(self, feats, labels, predict):

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long() # BATCH(=4) * 64 * 512
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1) # after shape: batch_size * (H * W * channel(channel maybe 1, because it's logits))
        predict = predict.contiguous().view(batch_size, -1) # after shape: batch_size * (H * W * channel(channel maybe 1, because it's logits))
        feats = feats.permute(0, 2, 3, 1) # after shape: batch_size * H * W * channels( = 256 )
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) #after shape: batch_size * (H*W) * channels ( = 256 )

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        # feats_, labels_ = self._hard_anchor_sampling_V2(feats, labels, predict) #221017 modify!
        # self._minor_anchor_sampling_(feats, labels, predict)
        #additional training & loss

        loss = self._contrastive(feats_, labels_)
        return loss


    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1] # in this case, (batch_size, 256)
        # X: feats  || batch_size * H * W * channels( = 256 )
        # y_hat: label( = ground truth ) ||
        # y: predict ( = predictions )
        # X: shape

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)

            # if gt label data has not information about some labels ( such as bycycle, bicyclist, bus, motorcycle ... )
            # we don't consider that label.

            this_classes = [x for x in this_classes if x not in self.ignore_label]
            # also, we reject ignore_label
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # how about to add codes, about excluding some indexes such as split the code min_views to x_views ?
            # because we set the max
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        # max_samples now: 1024, total_classes: batch size * (one batch image classes)
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1) # X_ == all batch class number * sampled_indices number * feature space
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_


    def _hard_anchor_sampling_V2(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]  # in this case, (batch_size, 256)
        # X: feats  || batch_size * H * W * channels( = 256 )
        # y_hat: label( = ground truth ) ||
        # y: predict ( = predictions )
        # X: shape

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # if gt label data has not information about some labels ( such as bycycle, bicyclist, bus, motorcycle ... )
            # we don't consider that label.

            this_classes = [x for x in this_classes if x not in self.ignore_label]
            # also, we reject ignore_label
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # how about to add codes, about excluding some indexes such as split the code min_views to x_views ?
            # because we set the max
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        # max_samples now: 1024, total_classes: batch size * (one batch image classes)
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)

                origin_hard_indices = hard_indices
                origin_easy_indices = easy_indices

                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)  # X_ == all batch class number * sampled_indices number * feature space
                y_[X_ptr] = cls_id
                X_ptr += 1

                if self.queue is not None:
                    remain_n_view = self.max_views - n_view

                    # num_easy = origin_easy_indices.shape[0]
                    # num_hard = origin_hard_indices.shape[0]
                    #
                    # if num_hard < remain_n_view // 2:
                    #     num_hard_keep = num_hard
                    #     num_easy_keep = remain_n_view - num_hard
                    #
                    # origin_hard_indices = origin_hard_indices[perm[:num_hard_keep]]
                    # origin_easy_indices = origin_easy_indices[perm[:num_easy_keep]]
                    # indices = torch.cat((origin_hard_indices, origin_easy_indices), dim=0)
                    #
                    if cls_id.item() not in self.queue:
                        # self.queue[cls_id.item()] = deque([X[ii, indices, :].squeeze(1)], maxlen = 100)
                        self.queue[cls_id.item()] = X[ii, indices, :].squeeze(1)
                    else:
                        tmp = self.queue[cls_id.item()]
                        tmp = torch.cat((tmp, X[ii, indices, :].squeeze(1)), dim=0)

                        if tmp.shape[0] > 100:
                            perm = torch.randperm(tmp.shape[0])
                            tmp = tmp[perm[:100],:]

                        self.queue[cls_id.item()] = tmp

                self._memory_bank_sampling()
        return X_, y_

    def _minor_anchor_sampling_(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]  # in this case, (batch_size, 256)
        # X: feats  || batch_size * H * W * channels( = 256 )
        # y_hat: label( = ground truth ) ||
        # y: predict ( = predictions )
        # X: shape

        classes = []
        total_classes = 0
        total_sample_num = 0

        for ii in range(batch_size):
            this_y = y_hat[ii]
            sorted_classes, num_class = torch.unique(this_y, return_counts=True)

            ratio_class = num_class.float() / torch.sum(num_class , dtype=torch.float32)
            order, arg_ = torch.sort(ratio_class)
            order = order < 0.04 # something local...need to change
            total_sample_num += torch.sum(num_class[arg_][order])
            # if gt label data has not information about some labels ( such as bycycle, bicyclist, bus, motorcycle ... )
            # we don't consider that label.

            # topk = order.shape[0] // 2
            # if topk < 1:
            #     topk = order.shape[0]

            this_classes = [x for x in sorted_classes[arg_][order]]
            this_classes = [x for x in this_classes if x not in self.ignore_label]
            # also, we reject ignore_label
            # this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # how about to add codes, about excluding some indexes such as split the code min_views to x_views ?
            # because we set the max
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = total_sample_num # self.max_samples // total_classes
        # max_samples now: 1024, total_classes: batch size * (one batch image classes)
        # n_view = min(n_view, self.max_views)

        # X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        # y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0

        for ii in range(batch_size):
            this_y_hat = y_hat[ii] # gt
            this_y = y[ii] # prediction
            this_classes = classes[ii]

            for cls_id in this_classes:

                indices = (this_y_hat == cls_id).nonzero()
                # indices = indices[:n_view]

                if self.minor_queue is not None:

                    if cls_id.item() not in self.minor_queue:
                        self.minor_queue[cls_id.item()] = X[ii, indices, :].squeeze(1)
                    else:
                        tmp = self.minor_queue[cls_id.item()]
                        tmp = torch.cat((tmp, X[ii, indices, :].squeeze(1)), dim=0)

                        if tmp.shape[0] > 100:
                            perm = torch.randperm(tmp.shape[0])
                            tmp = tmp[perm[:100],:]

                        self.minor_queue[cls_id.item()] = tmp
        print("minor samples!")
    def _memory_bank_sampling(self):
        feats_keys = list(self.queue.keys())
        feats_minor_keys = list(self.minor_queue.keys())


    def _hard_anchor_sampling_V3(self, X, y_hat, y):
        # deprecated version
        batch_size, feat_dim = X.shape[0], X.shape[-1]  # in this case, (batch_size, feat_dim = 256)

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii] # ground truth label
            this_classes = torch.unique(this_y) # real included gt label, in one dataset

            this_classes = [x for x in this_classes if x not in self.ignore_label]
            class_pts = {}

            for x in this_classes:
                class_nums = (this_y == x).nonzero().shape[0]
                class_pts.update({x:class_nums})

            class_pts_sort = list(sorted(class_pts.items(), key=lambda x: x[1]))
            minor_classes = round(len(class_pts_sort) * 0.4)
            # 0.4 is the ratio of scarce points data, we need to parameterize this value in ARCH.yaml

            minor_labels = class_pts_sort[:minor_classes]
            this_classes = [x[0] for x in minor_labels]
            #### *** end *** ####


            # original codes
            # this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        # max_samples now: 1024, total_classes: batch size * (one batch image classes)
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                # delete it
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard < n_view / 2 and num_easy < n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = num_easy
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                if num_easy == 0:
                    perm = torch.randperm(num_hard)
                    indices = torch.tensor(hard_indices)

                    for _ in range(n_view - num_hard):
                        tperm = random.choice(perm)
                        indices = torch.cat((indices, indices[tperm].view(-1,1)), dim=0)

                elif num_hard == 0:
                    perm = torch.randperm(num_easy)
                    indices = torch.tensor(easy_indices)

                    for _ in range(n_view - num_easy):
                        tperm = random.choice(perm)
                        indices = torch.cat((indices, indices[tperm].view(-1,1)), dim=0)

                elif num_easy + num_hard < n_view:
                    perm = torch.randperm(num_hard)
                    indices = torch.tensor(hard_indices)

                    for _ in range(n_view - (num_easy + num_hard)):
                        tperm = random.choice(perm)
                        indices = torch.cat((indices, indices[tperm].view(-1, 1)), dim=0)

                    indices = torch.cat((indices, easy_indices), dim=0)

                else:
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                if len(indices) > n_view:
                    perm = torch.randperm(len(indices))
                    indices = indices[perm[:n_view]]

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        # anchor_num = class numbers all in batches ( overlap class can exist) # 37
        # n_view = indices number. static for each samples (hard + easy) # 27

        labels_ = labels_.contiguous().view(-1, 1) # gt == shape(37, )
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        # mask = one hot encoding map for n_view sampling data, shape (37 * 37)
        # mask == diagonal matrix ! ! ! ! Identity matrix!!!

        contrast_count = n_view # 27
        # before feats_ == anchor_num * n_view * embedding dimension

        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0) # shape(1015 * 256)
        # after contrast_feature == (anchor_num * n_view) * embedding dimension


        anchor_feature = contrast_feature # dimension 256 embedding space
        # embed 1 : anchor_num * n_view features, embed 2 : anchor_num * n_view features ...
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature) # in L_NCE, i*i+ / tau
        # anchor feature * contrast feature / 256 -> same label features have max values

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss