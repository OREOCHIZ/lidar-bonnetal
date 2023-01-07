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

from tasks.semantic.modules.segmentator_contrastive import *
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.modules.ioueval import *


class Confusion_Test():
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
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

    self.evaluator = iouEval(self.parser.get_n_classes(), self.device, [0]) # 0 == ignore class

  def infer(self):
    # do train set
    # self.infer_subset(loader=self.parser.get_train_set(),
    #                   to_orig_fn=self.parser.to_original)

    # do valid set
    self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)
    # # do test set
    # self.infer_subset(loader=self.parser.get_test_set(),
    #                   to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      add_batch_counter = 0
      for i, (proj_in, proj_mask, proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)

        path_seq = path_seq[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          proj_labels = proj_labels.cuda()


        # compute output
        proj_output = self.model(proj_in, proj_mask)

        seg_out = proj_output['seg']
        embed_out = proj_output['embed']
        self.evaluator.reset()

        proj_argmax = seg_out[0].argmax(dim=0)
        self.evaluator.addBatch_save_conf(proj_argmax, proj_labels, add_batch_counter)
        add_batch_counter += 1