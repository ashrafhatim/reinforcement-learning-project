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
  def __init__(self, env_name, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decrement=0.001, learning_rate=0.0001, batch_size=128,
               n_episodes = 700, n_steps = 5000, buffer_size = 100000, hid1_dim=200, hid2_dim=128, path=None, tb_path=None, device = 'cpu', printLog = False, saveFreq = 2 ):
    """
    The Agent Clsss.
    ---
    INPUTS:
      env_name: (str) the environment name.
      gamma: (float) bellman equation constant.
      epsilon: (float) epsilon greedy statring value.
      epsilon_min: (float) min epsilon value.
      epsilon_decrement: (float) epsilon decrement.
      learning_rate: (float) learning rate.
      batch_size: (int) batch size.
      n_episodes: (int) number of games.
      n_steps: (int) max number of steps per episode.
      buffer_size: (int) max buffer size.
      hid1_dim: (int) first hidden dimension.
      hid2_dim: (int) second hidden dimension.
      path: (str) path for the saving directory.
      tb_path: (str) path for the tensorboard directory.
      device: (str) the available device.
      printLog: (bool) true to print the log while training.

    """
    self.device = device
    self.env_name = env_name
    self.env = gym.make(self.env_name)
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
    self.path = path
    self.tb_path = tb_path
    self.printLog = printLog
    self.saveFreq = saveFreq
    # inialise tensorboard writer
    self.sw = SummaryWriter(self.tb_path)
    # initialise the buffer
    self.buffer = ReplayBuffer(self.buffer_size,[self.state_dims])
    # initialise the networks
    self.dqn = DQN(self.state_dims, self.hid1_dim, self.hid2_dim, self.env.action_space.n).to(self.device)
    self.target_net = DQN(self.state_dims, self.hid1_dim, self.hid2_dim, self.env.action_space.n).to(self.device)
    self.target_net.load_state_dict(self.dqn.state_dict())
    self.target_net.eval()
    # initialise the loss
    self.loss_fn = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
    # list to save the reward per epoch
    self.rewards = [] 
    self.periodic_reward = 0

  def epsilon_greedy_action(self, state, env):
    """
    Epsilon greedy function.
    explore or learn depending, depend on epsilon.
    ---
    INPUT
      state: the current state.
      env: the environment.
    OUTPUT
      action: the current action.
    """
    explore = np.random.uniform() < self.epsilon
    if explore:
      action = env.action_space.sample()
    else:
      state = torch.tensor([ state ]).to(self.device)
      actions = self.dqn.forward(state)
      action = torch.argmax(actions).item()
    return action
    
  def step(self, state, env):
    """
    One step of gradient update.
    ---
    INPUTS
      state: the current state.
      env: the environment.
    OUTPUTS
      next_state: next state.
      env: the environment.
      done: (bool) true if the episode ended.

    """
    # do one env step and add the results to the memory
    self.dqn.eval()
    action = self.epsilon_greedy_action(state, env)
    next_state, reward, done, _ = env.step(action)
    self.buffer.push(state,action,reward,next_state,done)
    self.dqn.train()
    # calculate the accomulated reward per episode
    self.periodic_reward += reward

    # clear the gradient graph  
    self.optimizer.zero_grad()
    # sample one batch form the memory
    state_batch, action_batch, reward_batch, next_state_batch, termination_batch = self.buffer.sample(self.batch_size, self.device)
    state_batch, action_batch, reward_batch, next_state_batch, termination_batch = state_batch, action_batch, reward_batch, next_state_batch, termination_batch
    # one step of bellman equation
    q_eval = self.dqn(state_batch)[np.arange(self.batch_size),action_batch.long()] 
    q_next = self.target_net(next_state_batch).detach()
    q_next[termination_batch] = 0.0 # for terminal y = r
    target = reward_batch + self.gamma * q_next.max(dim=1)[0] 
    # calculate the loss
    loss = self.loss_fn(target,q_eval).to(self.device)
    loss.backward()
    # update the parameters
    self.optimizer.step() 

    # decrease epsilon or keep the minimum value
    if self.epsilon > self.epsilon_min:
      self.epsilon = self.epsilon - self.epsilon_decrement
    else: 
      self.epsilon = self.epsilon_min

    return next_state, env, done

  def episode(self, env):
    """
    One episode. And update the target network.
    ---
    INPUT
      env: the environment.
    """
    # reset the environment
    state = env.reset()
    done = False
    step = 0
  
    while not done and step < self.n_steps:
      state, env, done = self.step(state, env)
      step += 1

    # update the targer network
    self.target_net.load_state_dict(self.dqn.state_dict())


  def train(self):
    """
    The training function.
    """
    # make sure the size of the memory if larger than the batch size.
    state = self.env.reset()
    for _ in range(self.batch_size):
      self.dqn.eval()
      action = self.env.action_space.sample() #self.epsilon_greedy_action(state)
      next_state, reward, done, _ = self.env.step(action)
      self.buffer.push(state,action,reward,next_state,done)
      state = next_state

    # training loop
    for i in range(self.n_episodes):
      save_path = self.path + "/#" + str(i)
      self.env.reset() #gym.make(self.env_name)
      env = self.env
      if (i % self.saveFreq) == 0:
        env = Monitor(env, save_path, force=True, video_callable=lambda episode: True)
      self.episode(env)
      if self.printLog and len(self.rewards) >= 50:
        avg_reward = np.mean(self.rewards[-50:])
        print('EPISODE ', i+1 , 'reward %.2f' % self.periodic_reward, 'average reward %0.2f'% avg_reward, 'epssilon %0.2f' % self.epsilon)
        self.env.close()
        ### show_video(save_path)
        print()
      # update tensor board
      self.rewards.append(self.periodic_reward)
      self.sw.add_scalar('periodic reward per episode', self.periodic_reward, i)
      avg_reward = np.mean(self.rewards[-50:])
      self.sw.add_scalar('average reward per episode', avg_reward, i)
      self.sw.add_scalar('ebs_history per episode', self.epsilon, i)
      self.periodic_reward = 0
