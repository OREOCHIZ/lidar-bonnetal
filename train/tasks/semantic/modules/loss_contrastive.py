from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tasks.semantic.modules.border_extractor import borderExtracter
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
        seg_loss = self.seg_creterion(torch.log(outputs.clamp(min=1e-8)), labels)

        # 221215
        _, predict = torch.max(outputs, 1)
        contrast_loss = self.contrast_creterion(feats, labels, predict)

        if self.ARCH['contrastive']['with_embed'] is True:
            return seg_loss + self.loss_weight * contrast_loss

        return seg_loss + 0 * contrast_loss


class ContrastNLLoss(nn.Module):
    def __init__(self, ARCH, loss_w):
        super(ContrastNLLoss, self).__init__()
        self.ARCH = ARCH
        self.loss_weight = self.ARCH['contrastive']['loss_weight']

        self.seg_creterion = nn.NLLLoss(weight=loss_w) # loss existing
        self.contrast_creterion = PixelContrastLoss(self.ARCH) # added loss

    def forward(self, feats, outputs, labels):
        seg_loss = self.seg_creterion(outputs, labels)
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
        self.border_ext = borderExtracter(20, 0)

        if self.ARCH['contrastive']['with_memory']:
            self.queue = {}
            self.minor_queue = {}
            self.memory_size = self.ARCH['contrastive']['memory_size']
            # self.register_buffer("pixel_queue", torch.randn(num_classes, self.memory_size, dim))


    def forward(self, feats, labels, predict):
        # self.border_sampling(feats, labels, predict)
        loss = self.anchor_aware_sampling(feats, labels, predict)

        ### edit seoyeon ###
        # labels = labels.unsqueeze(1).float().clone()
        # labels = torch.nn.functional.interpolate(labels,
        #                                          (feats.shape[2], feats.shape[3]), mode='nearest')
        # labels = labels.squeeze(1).long() # BATCH(=4) * 64 * 512
        # assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        #
        # batch_size = feats.shape[0]
        #
        # labels = labels.contiguous().view(batch_size, -1) # after shape: batch_size * (H * W * channel(channel maybe 1, because it's logits))
        # predict = predict.contiguous().view(batch_size, -1) # after shape: batch_size * (H * W * channel(channel maybe 1, because it's logits))
        # feats = feats.permute(0, 2, 3, 1) # after shape: batch_size * H * W * channels( = 256 )
        # feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) #after shape: batch_size * (H*W) * channels ( = 256 )
        #
        # feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        #
        # loss = self._contrastive(feats_, labels_)
        ### edit seoyeon ###
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

    def border_sampling(self, X, y_hat, y):
        # X = b x C x H x W
        batch_size, feat_dim = X.shape[0], X.shape[1]
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_pred = y[ii]
            be_yhat = self.border_ext.get_border_from_label(this_y, erode_iter=1)
            # nonzero_mask = this_y > 0


            be_yhat_mask = ~ ((this_y == 0) | (be_yhat == 0))
            border_label = this_y[be_yhat_mask]

            #### idea... ####
            # unq, cnt = torch.unique(border_label, return_counts = True)
            # unq_ext = torch.where(cnt == 1)
            # if len(unq_ext[0]) != 0:
            #     for ext in unq_ext[0]:
            #         be_yhat_mask = (be_yhat_mask != ext)
            #     border_label = this_y[be_yhat_mask]
            # ### ###

             # (be_yhat * this_y).nonzero()


            import numpy as np
            # np.save("/ws/result/border_test_512/border_label_distribution.npy", border_label.detach().cpu().numpy())
            # which is easy border or hard border ????

            easy_border_mask = (torch.eq(this_y, this_pred)) & be_yhat_mask
            easy_b_label = this_y[easy_border_mask]
            # np.save("/ws/result/border_test_2048/easy_b_label.npy", easy_b_label.detach().cpu().numpy())

            hard_border_mask = (torch.ne(this_y, this_pred)) & be_yhat_mask
            hard_b_label = this_y[hard_border_mask]
            assert (border_label.shape[0] == (easy_b_label.shape[0] + hard_b_label.shape[0]))

            unq, cnt = torch.unique(hard_b_label, return_counts=True)
            unq_ext = torch.where(cnt == 1)
            if len(unq_ext[0]) != 0:
                for ext in unq_ext[0]:
                    hard_border_mask = (hard_border_mask & (this_y != unq[ext]))
                hard_b_label = this_y[hard_border_mask]
            ### ###


            border_class = torch.unique(border_label, return_counts=True)


            # hard_b_pred_label = this_pred[hard_border_mask]
            # np.save("/ws/result/border_test_2048/hard_b_label_gt.npy", hard_b_label.detach().cpu().numpy())
            # np.save("/ws/result/border_test_2048/hard_b_label_pred.npy", hard_b_pred_label.detach().cpu().numpy())

            # border_class = torch.unique(border_label)

            print(border_class)

            # border_feature = X[ii, :, be_yhat_mask]
            border_feature = X[ii, :, hard_border_mask]
            print(border_feature)

            # border_label = border_label.contiguous().view(-1, 1)
            hard_b_label = hard_b_label.contiguous().view(-1, 1)

            # contrast_mask = torch.eq(border_label, torch.transpose(border_label, 0, 1)).float().cuda() # border samples mask
            contrast_mask = torch.eq(hard_b_label, torch.transpose(hard_b_label, 0, 1)).float().cuda() # border samples mask

            border_dot = torch.div(torch.matmul(torch.transpose(border_feature, 0, 1), border_feature ),
                                            self.temperature)

            logits_max, _ = torch.max(border_dot, dim=1, keepdim=True)
            logits = border_dot - logits_max.detach()

            neg_mask = 1 - contrast_mask

            logits_mask = 1 - torch.eye(contrast_mask.shape[0])
            contrast_mask = contrast_mask * logits_mask.cuda()


            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits + neg_logits)
            # log_prob2 = torch.log(exp_logits / (exp_logits + neg_logits)) # there's no need to do

            mean_log_prob_pos = (contrast_mask * log_prob).sum(1) / contrast_mask.sum(1)  # good not good! if sample is one it goes to zero ...

            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
            print("hard border loss: ", loss)

            # import numpy as np
            # np.save("/ws/result/border_test_512/border_label.npy", border_label.detach().cpu().numpy())
            # np.save("/ws/result/border_test_512/contrast_mask.npy", contrast_mask.detach().cpu().numpy())
            # np.save("/ws/result/border_test_512/border_dot.npy", border_dot.detach().cpu().numpy())

            print("borrrrder")

    def anchor_aware_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[1]
        gt_class_all_batch = y_hat.contiguous().view(batch_size, -1)

        for ii in range(batch_size):
            gt_class, anch_num = torch.unique(gt_class_all_batch[ii], return_counts = True)
            this_y = y_hat[ii]
            this_pred = y[ii]
            total_loss = []
            for i, one_class in enumerate(gt_class):
                if one_class == 0 or anch_num[i] < 2:
                    continue
                else:
                    class_mask = (this_y == one_class) & (this_y > 0)

                    easy_positive_mask = class_mask & (this_y == this_pred)
                    hard_positive_mask = class_mask & (this_y != this_pred)
                    hard_negative_mask = (this_y > 0) & (this_y != one_class) & (this_pred == one_class)

                    ep_feat = None

                    if not torch.all(~easy_positive_mask):
                        ep_feat = X[ii, :, easy_positive_mask]
                    else:
                        continue

                    hp_feat = None
                    if not torch.all(~ hard_positive_mask):
                        hp_feat = X[ii, :, hard_positive_mask]
                    else:
                        continue
                    hn_feat = None

                    if not torch.all(~ hard_negative_mask):
                        hn_feat = X[ii, :, hard_negative_mask]
                    else:
                        continue

                    if (ep_feat == None) | (hp_feat == None) | (hn_feat == None):
                        continue

                    a, b, c = ep_feat.shape[1], hp_feat.shape[1], hn_feat.shape[1]

                    # pseudo code ...
                    a_keep = 0
                    b_keep = 0
                    c_keep = 0
                    if c > self.max_samples * 0.5:
                        c_keep = int(self.max_samples * 0.5)
                    else:
                        c_keep = c

                    if a > self.max_samples * 0.25 :
                        a_keep = int(self.max_samples * 0.25)
                    else:
                        a_keep = a

                    if b > self.max_samples * 0.25:
                        b_keep = int(self.max_samples * 0.25)
                    else:
                        b_keep = b

                    perm = torch.randperm(c_keep)
                    hn_feat = hn_feat[:, perm[:c_keep]]
                    perm = torch.randperm(b_keep)
                    hp_feat = hp_feat[:, perm[:b_keep]]
                    perm = torch.randperm(a_keep)
                    ep_feat = ep_feat[:, perm[:a_keep]]

                    pos_dot = torch.div(torch.matmul(torch.transpose(hp_feat, 0, 1), ep_feat), self.temperature) # i*i+ / tau, P * 256 * 256 * A == P*A
                    neg_dot = torch.div(torch.matmul(torch.transpose(hn_feat, 0, 1), ep_feat), self.temperature)  # i*i- / tau, N * 256 * 256 * A == N*A

                    # i * i can be +10 ~ -10
                    logit_max = 1 / self.temperature

                    pos_dot = pos_dot - logit_max
                    neg_dot = neg_dot - logit_max

                    exp_pos = torch.exp(pos_dot)
                    exp_neg = torch.exp(neg_dot)
                    exp_neg = exp_neg.sum(0, keepdim= True)
                    log_prob = pos_dot - torch.log(exp_pos + exp_neg) #problem
                    mean_log_prob = log_prob.sum(0) / log_prob.shape[0]
                    loss = - (self.temperature / self.base_temperature) * mean_log_prob
                    loss = loss.mean()
                    total_loss.append(loss)

                    # print("anchor aware sampling")
            anc_aware_loss = sum(total_loss) / len(total_loss)
            # print("anchor aware loss: ", anc_aware_loss)
            return anc_aware_loss

    def consine_similarity_test(self, X, y_hat, y):
        # X = b x C x H x W
        batch_size, feat_dim = X.shape[0], X.shape[1]
        gt_class_all_batch = y_hat.contiguous().view(batch_size, -1)


        for ii in range(batch_size):
            # class aware
            gt_class = torch.unique(gt_class_all_batch[ii]) # tensor returned
            this_y = y_hat[ii] # gt
            this_pred = y[ii] # prediction
            # be_yhat = self.border_ext.get_border_from_label(this_y, erode_iter=1)

            for one_class in gt_class:
                if one_class == 0:
                    continue
                class_mask = (this_y == one_class) & (this_y > 0)
                # class_feature = X[ii, :, class_mask]

                right_pred_mask = class_mask & (this_y == this_pred)
                easy_positive_mask = right_pred_mask

                feature = X[ii, :, easy_positive_mask] # easy positive_feature
                # label = this_y[easy_positive_mask]

                import numpy as np
                # np.save("/ws/result/cos_sim_test_2048/easy_positive_class_" + str(one_class.item()) + "_label.npy", label.detach().cpu().numpy())
                np.save("/ws/result/cos_sim_test_2048/easy_positive_class_"+ str(one_class.item())  +"_feature.npy", feature.cpu().numpy())


                hard_positive_mask = class_mask & (this_y != this_pred)
                feature = X[ii, :, hard_positive_mask]
                label = this_y[hard_positive_mask]

                np.save("/ws/result/cos_sim_test_2048/hard_positive_class_" + str(one_class.item()) + "_label.npy", label.detach().cpu().numpy())
                np.save("/ws/result/cos_sim_test_2048/hard_positive_class_"+ str(one_class.item())  +"_feature.npy", feature.detach().cpu().numpy())

                #
                easy_negative_mask = (this_y > 0) & (this_y != one_class) & torch.eq(this_y,this_pred)
                feature = X[ii, :, easy_negative_mask]
                # label =  this_y[easy_negative_mask]

                # np.save("/ws/result/cos_sim_test_2048/easy_negative_class_" + str(one_class.item()) + "_label.npy",
                #         label.detach().cpu().numpy())
                np.save("/ws/result/cos_sim_test_2048/easy_negative_class_" + str(one_class.item()) + "_feature.npy",
                        feature.detach().cpu().numpy())

                hard_negative_mask = (this_y > 0) & (this_y != one_class) & (this_pred == one_class)
                feature = X[ii, :, hard_negative_mask]
                label = this_y[hard_negative_mask]

                np.save("/ws/result/cos_sim_test_2048/hard_negative_class_" + str(one_class.item()) + "_label.npy", label.cpu().numpy())
                np.save("/ws/result/cos_sim_test_2048/hard_negative_class_" + str(one_class.item()) + "_feature.npy",feature.cpu().numpy())

            print("cos sim!!!!")



            # # border_label = border_label.contiguous().view(-1, 1)
            # hard_b_label = hard_b_label.contiguous().view(-1, 1)
            #
            # # contrast_mask = torch.eq(border_label, torch.transpose(border_label, 0, 1)).float().cuda() # border samples mask
            # contrast_mask = torch.eq(hard_b_label, torch.transpose(hard_b_label, 0, 1)).float().cuda() # border samples mask
            #
            # border_dot = torch.div(torch.matmul(torch.transpose(border_feature, 0, 1), border_feature ), self.temperature)
            #
            # logits_max, _ = torch.max(border_dot, dim=1, keepdim=True)
            # logits = border_dot - logits_max.detach()
            #
            # neg_mask = 1 - contrast_mask
            #
            # logits_mask = 1 - torch.eye(contrast_mask.shape[0])
            # contrast_mask = contrast_mask * logits_mask.cuda()
            #
            #
            # neg_logits = torch.exp(logits) * neg_mask
            # neg_logits = neg_logits.sum(1, keepdim=True)
            #
            # exp_logits = torch.exp(logits)
            # log_prob = logits - torch.log(exp_logits + neg_logits)
            # # log_prob2 = torch.log(exp_logits / (exp_logits + neg_logits)) # there's no need to do
            #
            # mean_log_prob_pos = (contrast_mask * log_prob).sum(1) / contrast_mask.sum(1)  # good not good! if sample is one it goes to zero ...
            #
            # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            # loss = loss.mean()
            # print("hard border loss: ", loss)
            #
            # # import numpy as np
            # # np.save("/ws/result/border_test_512/border_label.npy", border_label.detach().cpu().numpy())
            # # np.save("/ws/result/border_test_512/contrast_mask.npy", contrast_mask.detach().cpu().numpy())
            # # np.save("/ws/result/border_test_512/border_dot.npy", border_dot.detach().cpu().numpy())

            print("borrrrder")



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
        minor_class = []
        major_class = []
        batch_size, feat_dim = X.shape[0], X.shape[-1]  # in this case, (batch_size, 256)

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
        import numpy as np # 221229

        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        # anchor_num = class numbers all in batches ( overlap class can exist) # 37
        # n_view = indices number. static for each samples (hard + easy) # 27

        labels_ = labels_.contiguous().view(-1, 1) # gt == shape(37, )
        # labels_npy = labels_.detach().cpu().numpy() # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/labels_.npy", labels_npy)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        # mask_npy = mask.detach().cpu().numpy()  # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/mask_before_repeat.npy", mask_npy)
        #not always to be diagonal matrix
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
        # anchor_dot_contrast_npy = anchor_dot_contrast.detach().cpu().numpy()  # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/anchor_dot_contrast.npy",anchor_dot_contrast_npy)
        # no
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # logits_npy = logits.detach().cpu().numpy()  # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/logits.npy", logits_npy)
        # #

        mask = mask.repeat(anchor_count, contrast_count)
        # mask_npy = mask.detach().cpu().numpy() # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/mask_after_repeat.npy", mask_npy)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask
        # mask_npy = mask.detach().cpu().numpy()  # 221229
        # np.save(
        #     "/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/mask_after_repeat2.npy",
        #     mask_npy)

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        # logits_npy = neg_logits.detach().cpu().numpy()  # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/neg_logits.npy", logits_npy)
        #

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits) # real strange
        #log_prob = logits  - torch.log(neg_logits) is enough?


        # log_prob_npy = log_prob.detach().cpu().numpy() # 221229
        # np.save("/ws/result/log_xentropy_contrastive_lr_3e-3_lr_decay_99e-3_RV_2048_xentopy_contrastive_heatmap/log_prob.npy", log_prob_npy)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # good

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss