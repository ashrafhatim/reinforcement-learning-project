import torch
import numpy as np


class ReplayBuffer:
  def __init__(self,buffer_size,state_dims,n_actions=2):
    self.state_dims = state_dims
    self.buffer_size = buffer_size
    self.counter = 0

    self.state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    self.next_state_buffer = np.zeros((buffer_size,*self.state_dims), dtype=np.float32)
    # there are now two actions so we adapt the action buffer to accomodate that
    self.action_buffer = np.zeros((buffer_size,n_actions), dtype=np.float32)
    self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
    self.termination_buffer = np.zeros(buffer_size, dtype=np.bool)

  def push(self, state, action, reward, next_state, done):
    index = self.counter % self.buffer_size

    self.state_buffer[index] = state
    self.next_state_buffer[index] = next_state
    self.action_buffer[index] = action
    self.reward_buffer[index] = reward
    self.termination_buffer[index] = done

    self.counter += 1

  def sample(self,batch_size):
    filled_buffer = min(self.buffer_size, self.counter)
    batch_idx = np.random.choice(filled_buffer, batch_size, replace=False)

    return torch.tensor(self.state_buffer[batch_idx]), torch.tensor(self.action_buffer[batch_idx]), torch.tensor(self.reward_buffer[batch_idx]), torch.tensor(self.next_state_buffer[batch_idx]), torch.tensor(self.termination_buffer[batch_idx])

  def __len__(self):
    return self.buffer_size