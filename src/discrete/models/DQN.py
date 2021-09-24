import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd


class DQN(nn.Module):
  def __init__(self, input_dim, hid1_dim, hid2_dim, n_actions):
    super(DQN, self).__init__()
    """
    The DQN network Class.
    ---
    INPUTS:
      input_dim: (int) the state dimensionss.
      hid1_dim: (int) the first hidden dimension.
      hid2_dim: (int) the second hidden dimension.
      n_actions: (int) the action dimensions.
    """
    self.input_dim = input_dim
    self.hid1_dim = hid1_dim
    self.hid2_dim = hid2_dim
    self.n_actions = n_actions
    self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hid1_dim),
            nn.ReLU(),
            nn.Linear(self.hid1_dim, self.hid2_dim),
            nn.ReLU(),
            nn.Linear(self.hid2_dim, self.n_actions)
        )

  def forward(self, state):
    return self.net(state.float())