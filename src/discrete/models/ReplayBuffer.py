import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd


class ReplayBuffer:
  def __init__(self,buffer_size,state_dims):
    """
    The Replay Buffer Class. 
    First in first out buffer, used as memory.
    Divided into 5 different buffers:
      1- state_buffer: (numpy.ndarray) the current states buffer.
      2- next_state_buffer: (numpy.ndarray) the next states buffer.
      3- action_buffer: (numpy.ndarray) the actions buffer.
      4- reward_buffer: (numpy.ndarray) the reward buffer.
      5- termination_buffer: (numpy.ndarray) the termination buffer. Used to track with end of episods actions.
    ---
    INPUTS
      buffer_size: (int) the buffer max size.
      state_dims: (int) the state dimensions.
    """
    self.state_dims = state_dims
    self.buffer_size = buffer_size
    # counter to keep track of the buffer index
    self.counter = 0
    # initialise the buffers
    self.state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    self.next_state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    self.action_buffer = np.zeros(buffer_size, dtype=np.float32)
    self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
    self.termination_buffer = np.zeros(buffer_size, dtype=np.bool)

  def push(self, state, action, reward, next_state, done):
    """
    Push state, action, reward, next_state, done into the reply buffer
    ---
    state, action, reward, next_state, done: (numpy.ndarray) 
    """
    # mode buffer_size to reset the index when exceed the max size.
    index = self.counter % self.buffer_size
    # fill the buffers
    self.state_buffer[index] = state
    self.next_state_buffer[index] = next_state
    self.action_buffer[index] = action
    self.reward_buffer[index] = reward
    self.termination_buffer[index] = done
    # increase the conter
    self.counter += 1

  def sample(self,batch_size, device):
    """
    Sample batch of inputs.
    ---
    batch_size: (int) the batch size.
    device: (str) the available device.
    """
    filled_buffer = min(self.buffer_size, self.counter)
    batch_idx = np.random.choice(filled_buffer, batch_size, replace=False)

    # convert to tensor
    state = torch.tensor(self.state_buffer[batch_idx]).to(device)
    actions = torch.tensor(self.action_buffer[batch_idx]).to(device)
    rewards = torch.tensor(self.reward_buffer[batch_idx]).to(device)
    next_states = torch.tensor(self.next_state_buffer[batch_idx]).to(device)
    terminations = torch.tensor(self.termination_buffer[batch_idx]).to(device)

    return state, actions, rewards, next_states, terminations

  def __len__(self):
    return self.buffer_size