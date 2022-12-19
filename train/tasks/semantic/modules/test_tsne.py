#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from tasks.semantic.modules.segmentator_contrastive import *
from tasks.semantic.postproc.KNN import KNN


class Test_TSNE():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TSNE in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    # do train set
    self.infer_subset(loader=self.parser.get_train_set(),
                      to_orig_fn=self.parser.to_original, mode= "train")
    # do valid set
    self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original, mode= "valid")
    # do test set
    self.infer_subset(loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original, mode = "test")

    print('Finished tsne')

    return

  def infer_subset(self, loader, to_orig_fn, mode="default"):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()


      for i, (proj_in, proj_mask, proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
          # first cut to rela size (batch size one allows it)
          print("path_name: ", path_name[0][0:6])
          p_x = p_x[0, :npoints]
          p_y = p_y[0, :npoints]
          proj_range = proj_range[0, :npoints]
          unproj_range = unproj_range[0, :npoints]
          path_seq = path_seq[0]
          path_name = path_name[0]

          if self.gpu:
              proj_in = proj_in.cuda()
              proj_mask = proj_mask.cuda()
              p_x = p_x.cuda()
              p_y = p_y.cuda()
              if self.post:
                  proj_range = proj_range.cuda()
                  unproj_range = unproj_range.cuda()

          # compute output
          print("here. . .")

          proj_output = self.model(proj_in, proj_mask)

          # seg_out = proj_output['seg'] #
          embed_out = proj_output['embed']

          batch_size = embed_out.size(0) #B
          embed_2d = embed_out.permute(0, 2, 3, 1) # B x H x W x C (=256)
          feature_size = embed_2d.size(-1) # 256
          embed_2d = embed_2d.view(-1, feature_size) # (B x H x W) X C

          encoded = embed_2d.cpu().detach().numpy()
          print("encoded : ", encoded.shape)
          Y = proj_labels.view(-1) # (H x W) X C, Batch == 1
          Y = Y.cpu().detach().numpy()

          tsne = TSNE()
          X_train_2D = tsne.fit_transform(encoded)
          X_train_2D = (X_train_2D - X_train_2D.min()) / (X_train_2D.max() - X_train_2D.min())
          print(X_train_2D.shape)

          print("Y.shape: ", Y.shape)
          labels = np.unique(Y)
          labels = [str(label) for label in labels]

          # plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=Y, s=2, cmap="tab10")
          plt.axis("off")
          scatter = plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=Y, s=2, cmap="gist_rainbow")
          handles, _ = scatter.legend_elements(prop='colors')
          plt.legend(handles, labels)

          path_fullname = "/ws/RANGENET_TSNE/post_" + mode + str(i) +".png"
          plt.savefig(path_fullname)
          break
          '''
          for idx, feats in enumerate(Y):
            plt.scatter(X_train_2D[idx, 0], X_train_2D[idx, 1], s=2, )
          '''
          # 3D
          # tsne = TSNE(n_components=3)
          # X_train_2D = tsne.fit_transform(encoded)
          # X_train_2D = (X_train_2D - X_train_2D.min()) / (X_train_2D.max() - X_train_2D.min())
          # print("X_train_2D: ", X_train_2D.shape)

          # print("Y.shape: ", Y.shape)
          #
          # plt.scatter3D(X_train_2D[:, 0], X_train_2D[:, 1], X_train_2D[:, 2], c=Y, s=2, cmap="tab10")
          # plt.axis("off")
          # path_fullname = "/ws/RANGENET_TSNE/post_" + str(i) + ".png"
          # plt.savefig(path_fullname)


