#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from dataloader import Dataloader
from model import Model


def main():
  # input image shape
  shape = (25, 25)
  # dataloader
  d = Dataloader(shape, batch_size=64)

  units = [shape[0] * shape[1], 20, 10]
  model = Model(units)
  # set model to training mode
  model.train()

  # max iterations
  max_iters = 4000
  # learning rate
  learning_rate = 0.01
  for i, batch in enumerate(iter(d)):
    # train the network
    x, y = batch
    # reshape the image to vector
    x = x.reshape(x.shape[0], -1)
    # normalize and convert to float32
    x = x.astype(np.float32) / 255
    # convert y to one-hot
    y = np.eye(10, dtype=np.float32)[y.flat]
    # decay learning rate
    learning_rate *= 0.9999
    pred = model(x, y, learning_rate=learning_rate)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
    print(f"iter: {i}, lr: {learning_rate:.4f}, acc: {acc:.2%}")
    if i >= max_iters:
      break
  model.save("./model.pkl")

  # test the network
  print("start testing")
  # set model to evaluation mode
  model.eval()
  test_batchs = 500
  acc = 0
  for i, batch in enumerate(iter(d)):
    x, y = batch
    x = x.reshape(x.shape[0], -1)
    x = x.astype(np.float32) / 255
    y = np.eye(10, dtype=np.float32)[y.flat]
    pred = model(x, y)
    acc += np.sum(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
    if i >= test_batchs:
      break
  acc /= (test_batchs * d.batch_size)
  print(f"test acc: {acc:.2%}")


if __name__ == "__main__":
  main()
