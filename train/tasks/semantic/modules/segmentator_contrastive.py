#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF
from tasks.semantic.modules.projection import ProjectionHead
import __init__ as booger


class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path = path
    self.path_append = path_append
    self.strict = False

    # get the model
    bboneModule = imp.load_source("bboneModule",
                                  booger.TRAIN_PATH + '/backbones_contrastive/' +
                                  self.ARCH["backbone"]["name"] + '.py')
    self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1,
                        self.backbone.get_input_depth(),
                        self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                        self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule",
                                    booger.TRAIN_PATH + '/tasks/semantic/decoders/' +
                                    self.ARCH["decoder"]["name"] + '.py')
    self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                         stub_skips=stub_skips,
                                         OS=self.ARCH["backbone"]["OS"],
                                         feature_depth=self.backbone.get_last_depth())

    self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(self.decoder.get_last_depth(),
                                        self.nclasses, kernel_size=3,
                                        stride=1, padding=1))
    self.proj_head = ProjectionHead(self.backbone.get_last_depth(), proj_dim=self.ARCH["contrastive"]["proj_dim"])


    if self.ARCH["post"]["CRF"]["use"]:
      print("use CRF")
      self.CRF = CRF_(self.ARCH["post"]["CRF"]["params"], self.nclasses)


    else:
      self.CRF = None

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # train CRF?
    if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
      for w in self.CRF.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    weights_proj_head = sum(p.numel() for p in self.proj_head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    print("Param projection head ", weights_proj_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/segmentation_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # 221004
      try:
        w_dict = torch.load(path + "/proj_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.proj_head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model proj head weights")
      except Exception as e:
        print("Couldn't load proj head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try CRF
      if self.CRF:
        try:
          w_dict = torch.load(path + "/segmentation_CRF" + path_append,
                              map_location=lambda storage, loc: storage)
          self.CRF.load_state_dict(w_dict, strict=True)
          print("Successfully loaded model CRF weights")
        except Exception as e:
          print("Couldn't load CRF, using random weights. Error: ", e)
          if strict:
            print("I'm in strict mode and failure to load weights blows me up :)")
            raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    y, skips = self.backbone(x) # 1024 * 64 * 64

    y_dec = y
    y_dec = self.decoder(y_dec, skips) # 30 * 64 * 64

    cls_out_ = self.head(y_dec)
    cls_out = F.softmax(cls_out_, dim=1)
    # cls_log_out = F.log_softmax(cls_out_, dim=1)


    _, _, h, w = y_dec.size()
    y_enc = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
    # now I interpolate encoded image with
    embed = self.proj_head(y_enc)

    if self.CRF:
      assert(mask is not None)
      cls_out_ = self.CRF(x, cls_out, mask)
      return {'seg': cls_out_, 'embed': embed}

    return {'seg': cls_out, 'embed': embed}

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    torch.save(self.decoder.state_dict(), logdir +
               "/segmentation_decoder" + suffix)
    torch.save(self.head.state_dict(), logdir +
               "/segmentation_head" + suffix)
    torch.save(self.proj_head.state_dict(), logdir +
               "/proj_head" + suffix) #221004
    if self.CRF:
      torch.save(self.CRF.state_dict(), logdir +
                 "/segmentation_CRF" + suffix)


