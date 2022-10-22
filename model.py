# -*- coding: utf-8 -*-
import pickle as pkl
from typing import List

import numpy as np


def sigmoid(x) -> np.ndarray:
  """Sigmoid function."""
  return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
  """Derivative of sigmoid function."""
  return sigmoid(x) * (1 - sigmoid(x))


def truncated_normal(shape, mean=0.0, stddev=1.0) -> np.ndarray:
  """Truncated normal distribution. Used for initializing weights."""
  return np.clip(
      np.random.normal(mean, stddev, shape), -2 * stddev, 2 * stddev)


class Model:
  """A simple neural network model.

  Parameters
  ----------
  units : List[int]
    The number of units in each layer.
  """

  def __init__(self, units: List[int]) -> None:
    self.units = units
    self.weights = []
    self.biases = []
    # init weights
    for h, w in zip(units[:-1], units[1:]):
      self.weights.append(truncated_normal((h, w)))
      self.biases.append(truncated_normal((w,)))
    self._train = True

  def train(self) -> None:
    """Set the model to training mode."""
    self._train = True

  def eval(self) -> None:
    """Set the model to evaluation mode."""
    self._train = False

  def __call__(self, x, y=None, learning_rate=0.01) -> np.ndarray:
    """Inference or training.

    Parameters
    ----------
    x : np.ndarray
      Input data. shape: (batch_size, input_size)
    y : np.ndarray, optional
      Target data. shape: (batch_size, output_size)

    Returns
    -------
    np.ndarray
      Prediction. shape: (batch_size, output_size)
    """
    # evaluate
    if not self._train:
      for w, b in zip(self.weights, self.biases):
        x = sigmoid(np.dot(x, w) + b)
      return x

    assert y is not None, "y must be provided in training mode."

    # === forward pass ===
    # z_s is collection of ``z`` in each layer
    z_s = []
    # a_s is collection of ``a`` in each layer
    a_s = [x]
    for w, b in zip(self.weights, self.biases):
      x = np.dot(x, w) + b
      z_s.append(x)
      x = sigmoid(x)
      a_s.append(x)

    # === backward pass ===
    # predefine: dL/dw
    delta_w = [0] * len(self.weights)
    # predefine: dL/db
    delta_b = [0] * len(self.biases)

    # calculate delta for the last layer
    # dL/dz = dL/da * da/dz
    delta = (a_s[-1] - y) * sigmoid_derivative(z_s[-1])
    # dL/dw = dL/dz * dz/dw
    delta_w[-1] = np.dot(a_s[-2].T, delta)
    # dL/db = dL/dz * dz/db
    delta_b[-1] = np.dot(np.ones((delta.shape[0], 1)).T, delta).flatten()

    # calculate delta for the hidden layers
    for layer in range(2, len(self.units)):
      # dL/dz = dL/da * da/dz
      delta = np.dot(delta, self.weights[-layer + 1].T) * sigmoid_derivative(z_s[-layer])
      # dL/dw = dL/dz * dz/dw
      delta_w[-layer] = np.dot(a_s[-layer - 1].T, delta)
      # dL/db = dL/dz * dz/db
      delta_b[-layer] = np.dot(np.ones((delta.shape[0], 1)).T, delta).flatten()

    # update weights and biases
    lr = learning_rate
    # w = w - lr * dL/dw
    self.weights = [w - lr * dw for w, dw in zip(self.weights, delta_w)]
    # b = b - lr * dL/db
    self.biases = [b - lr * db for b, db in zip(self.biases, delta_b)]

    return x

  def save(self, path: str) -> None:
    """Save the model to a file."""
    state_dict = {
        "units": self.units,
        "weights": self.weights,
        "biases": self.biases,
    }
    with open(path, "wb") as fp:
      pkl.dump(state_dict, fp)
    print("model saved to:", path)

  @classmethod
  def load(cls, path: str) -> "Model":
    """Load the model from a file."""
    print("loading model from:", path)
    with open(path, "rb") as fp:
      state_dict = pkl.load(fp)
    model = cls(state_dict["units"])
    model.weights = state_dict["weights"]
    model.biases = state_dict["biases"]
    return model
