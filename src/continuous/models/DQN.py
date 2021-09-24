import torch.nn as nn

class DQN(nn.Module):
  def __init__(self, input_dim, hid1_dim, hid2_dim, n_actions):

    super(DQN, self).__init__()
  
    self.input_dim = input_dim
    self.hid1_dim = hid1_dim
    self.hid2_dim = hid2_dim
    self.n_actions = n_actions
    self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hid1_dim),
            nn.ReLU(),
            nn.Linear(self.hid1_dim, self.hid2_dim),
            nn.ReLU()
        )
    # instead of outputting one value we output the q values for the two actions
    self.last1 = nn.Linear(self.hid2_dim, self.n_actions)
    self.last2 = nn.Linear(self.hid2_dim, self.n_actions)

  def forward(self, state):
    x = self.net(state)
    # returns the two different sets of q values
    return self.last1(x), self.last2(x)