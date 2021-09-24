import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import torch
import torch.nn.functional as F 
from torch import optim
import numpy as np

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output

from pprint import pprint

from IPython.display import HTML

from torch.utils.tensorboard import SummaryWriter

from pyvirtualdisplay import Display

from models.Agent import *
from models.DQN import *
from models.ReplayBuffer import *


class Agent:
  def __init__(self, env, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decrement=0.001, learning_rate=0.001, 
               batch_size=64,n_episodes = 500, n_steps = 5000, buffer_size = 100000, hid1_dim=128, hid2_dim=128, path="", tb_path="", displayEnv=False):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.env = env
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decrement = epsilon_decrement
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.n_episodes = n_episodes
    self.n_steps = n_steps
    self.buffer_size = buffer_size
    self.hid1_dim = hid1_dim
    self.hid2_dim = hid2_dim
    self.state_dims = self.env.observation_space.shape[0]

    self.buffer = ReplayBuffer(self.buffer_size,[self.state_dims])

    self.dqn = DQN(self.state_dims, self.hid1_dim, self.hid2_dim, 21).to(self.device)
    self.target_net = DQN(self.state_dims, self.hid1_dim, self.hid2_dim, 21).to(self.device).eval()
    self.target_net.load_state_dict(self.dqn.state_dict())

    self.loss_fn = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

    self.rewards = []
    self.average_rewards = []
    self.ebs_history = []
    self.periodic_reward = 0

    self.path = path
    self.tb_path = tb_path
    self.displayEnv = displayEnv
    self.sw = SummaryWriter(self.tb_path)

  def epsilon_greedy_action(self, state):
    explore = np.random.uniform() < self.epsilon
    if explore:
      action = self.env.action_space.sample()
    else:
      state = torch.tensor([ state ]).to(self.device)
      # we take the argmax which will give us an integer in [0,20] then divide it by 10 and substract 1 to get a float between -1 and +1
      action = np.array(list((output.argmax().item())/10 - 1 for output in  self.dqn.forward(state)))
    return action
    
  def step(self, state):
    self.dqn.eval()
    action = self.epsilon_greedy_action(state)
    next_state, reward, done, _ = self.env.step(action)
    self.buffer.push(state,((action+1)*10).round(),reward,next_state,done)
    self.dqn.train()

    self.periodic_reward += reward

    self.optimizer.zero_grad()

    state_batch, action_batch, reward_batch, next_state_batch, termination_batch = self.buffer.sample(self.batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, termination_batch = state_batch.to(self.device), action_batch.to(self.device), reward_batch.to(self.device), next_state_batch.to(self.device), termination_batch.to(self.device)

    # list of q values for each action
    q_eval_list = self.dqn(state_batch)
    q_eval = [
        q_eval_list[0][np.arange(self.batch_size),action_batch[:,0].long()],
        q_eval_list[1][np.arange(self.batch_size),action_batch[:,1].long()]
    ]
    
    q_next_list = self.target_net(next_state_batch) 
    q_next_list[0][termination_batch] = 0.0 # for terminal y = r
    q_next_list[1][termination_batch] = 0.0 # for terminal y = r
    # we compute the target for both actions
    target = [
        reward_batch + self.gamma * q_next_list[0].max(dim=1)[0],
        reward_batch + self.gamma * q_next_list[1].max(dim=1)[0],
    ]
    # we sum the losses to be able to backpropagate on both
    loss = self.loss_fn(target[0],q_eval[0]).to(self.device) + self.loss_fn(target[1],q_eval[1]).to(self.device)

    loss.backward()
    self.optimizer.step() 

    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon - self.epsilon_decrement
    else: 
      self.epsilon = self.epsilon_min

    return next_state, done

  def episode(self):
    state = self.env.reset()
    done = False
    step = 0

    while not done and step < self.n_steps:
      state, done = self.step(state)
      step += 1

    self.target_net.load_state_dict(self.dqn.state_dict())

  def train(self):
    state = self.env.reset()
    for _ in range(self.batch_size):
      self.dqn.eval()
      action = self.epsilon_greedy_action(state)
      next_state, reward, done, _ = self.env.step(action)
      # we multiply the actions by 10 after adding 1 and round it to have integers between 0 and 20 to use for our classification loss
      self.buffer.push(state,((action+1)*10).round(),reward,next_state,done)
      state = next_state

    for i in range(self.n_episodes): 
      savePath = self.path + "/#"+str(i)
      if ( (i + 1) % 50 == 0 or i==0 or (i+1)==self.n_episodes ) and self.displayEnv: self.env = Monitor(self.env, savePath, force=True, video_callable=lambda episode: True)
      self.episode()
      if ( (i + 1) % 50 == 0 or i==0 or (i+1)==self.n_episodes ) and len(self.rewards) >= 50:
        avg_reward = np.mean(self.rewards[-50:])
        self.average_rewards.append((avg_reward,i))
        self.sw.add_scalar("Average reward per episode",avg_reward,i)
        print('EPISODE ', i+1 , 'reward %.2f' % self.periodic_reward, 'average reward %0.2f'% avg_reward, 'epssilon %0.2f' % self.epsilon)
        self.env.close()
        # show_video(savePath)
        print()
      self.rewards.append(self.periodic_reward)
      self.sw.add_scalar("Reward per episode",self.periodic_reward,i)
      self.ebs_history.append(self.epsilon)
      self.sw.add_scalar("Epsilon per episode",self.epsilon,i)
      self.periodic_reward = 0