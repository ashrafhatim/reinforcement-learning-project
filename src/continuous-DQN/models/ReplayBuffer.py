import torch
import numpy as np

class ReplayBuffer:
  def __init__(self,buffer_size,state_dims,n_actions=2):
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
      n_actions: (int) the number of actions
    """
    self.state_dims = state_dims
    self.buffer_size = buffer_size
    self.counter = 0

    self.state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    self.next_state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    self.action_buffer = np.zeros((buffer_size,n_actions), dtype=np.float32)
    self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
    self.termination_buffer = np.zeros(buffer_size, dtype=np.bool)

  def push(self, state, action, reward, next_state, done):
    """
    Push state, action, reward, next_state, done into the reply buffer
    ---
    state, action, reward, next_state, done: (numpy.ndarray) 
    """
    index = self.counter % self.buffer_size

    self.state_buffer[index] = state
    self.next_state_buffer[index] = next_state
    self.action_buffer[index] = action
    self.reward_buffer[index] = reward
    self.termination_buffer[index] = done

    self.counter += 1

  def sample(self,batch_size):
    """
    Sample batch of inputs.
    ---
    batch_size: (int) the batch size.
    device: (str) the available device.
    """
    filled_buffer = min(self.buffer_size, self.counter)
    batch_idx = np.random.choice(filled_buffer, batch_size, replace=False)

    return torch.tensor(self.state_buffer[batch_idx]), torch.tensor(self.action_buffer[batch_idx]), torch.tensor(self.reward_buffer[batch_idx]), torch.tensor(self.next_state_buffer[batch_idx]), torch.tensor(self.termination_buffer[batch_idx])

  def __len__(self):
    return self.buffer_size