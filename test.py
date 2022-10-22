#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

from model import Model


def test():
  files = [
      os.path.join("examples", x)
      for x in os.listdir("./examples") if x.endswith(".png")
  ]
  files.sort()
  model = Model.load("./model.pkl")
  model.eval()
  for file in files:
    # read single channel image
    x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # add batch dimension
    x = x[np.newaxis, ...]
    # reshape the image to vector
    x = x.reshape(x.shape[0], -1)
    # normalize and convert to float32
    x = x.astype(np.float32) / 255
    # predict
    y = model(x)
    pred_cls = np.argmax(y, axis=1)
    print("image: {}, prediction: {}".format(file, pred_cls))


if __name__ == "__main__":
  test()
