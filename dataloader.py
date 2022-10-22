#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import string
import unittest
from typing import List, Tuple

import cv2
import numpy as np


class Dataloader:
  """Data loader for random images with digits.

  Parameters
  ----------
  shape : List[int]
      The shape of image, (height, width).
  batch_size : int
      The batch size of data loader.
  """

  def __init__(self, shape: List[int], batch_size: int) -> None:
    self.shape = shape
    self.batch_size = batch_size

  def __iter__(self) -> Tuple:
    init_shape = [100, 100]
    h, w = init_shape
    while True:
      digits = np.random.randint(
          0, 10, size=(self.batch_size, 1), dtype=np.int32)
      # suppose the image's channel is BGR
      images = np.zeros((self.batch_size, *init_shape), dtype=np.uint8)
      rimages = np.zeros((self.batch_size, *self.shape), dtype=np.uint8)

      font_size = 3
      # put text on image
      for idx in range(self.batch_size):
        cv2.putText(
            images[idx], str(digits.flat[idx]), (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size, 255, 10, cv2.LINE_AA)
        # random rotate, angle: -50 - 50
        angle = np.random.randint(-50, 50)
        M = cv2.getRotationMatrix2D((h/2, w/2), angle, 1)
        images[idx] = cv2.warpAffine(images[idx], M, (h, w))
        images[idx] = 255 - images[idx]
        # fill random noise, range: [150, 200]
        noise = np.random.randint(120, 255, size=(h, w), dtype=np.uint8)
        images[idx] = np.where(images[idx] == 255, noise, images[idx])
        # resize image
        rimages[idx] = cv2.resize(images[idx], self.shape)
      yield rimages, digits


def save_image_to_disk(batch: Tuple, save_dir: str = "./data") -> None:
  """Save images to disk for visualization.

  Parameters
  ----------
  batch : Tuple
      The batch of images and labels. Output of Dataloader.
  save_dir : str
      The directory to save images. Default is "./data".
  """
  os.makedirs(save_dir, exist_ok=True)
  images, labels = batch
  # save to disk, file name: save_dir + random string + label + ".png"
  for image, label in zip(images, labels.flat):
    random_string = "".join(
        random.choice(string.ascii_lowercase) for _ in range(10))
    file_name = f"{save_dir}/{random_string}_{label}.png"
    cv2.imwrite(file_name, image)


class Tester(unittest.TestCase):

  def test_dataloader(self):
    dataloader = Dataloader(shape=[25, 25], batch_size=32)
    for batch in dataloader:
      # visualize the image, save to disk
      save_image_to_disk(batch)
      break


if __name__ == "__main__":
  unittest.main()
