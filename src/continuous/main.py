import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import torch

from torch import optim

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output

import random, os.path
from pprint import pprint

from IPython.display import HTML

from torch.utils.tensorboard import SummaryWriter

from pyvirtualdisplay import Display

from models.Agent import *
from models.DQN import *
from models.ReplayBuffer import *

# import Agent
# from models.DQN import DQN
# from models.ReplayBuffer import ReplayBuffer
from utils import helper
import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # env, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decrement=0.001, learning_rate=0.001, 
    #            batch_size=64,n_episodes = 500, n_steps = 5000, buffer_size = 100000, hid1_dim=128, hid2_dim=128, path="", tb_path=""
      
    parser.add_argument("--seed", default= 0, type= int)
    parser.add_argument("--gamma", default= 0.99, type= float)
    parser.add_argument("--epsilon", default= 1, type= float)
    parser.add_argument("--epsilon-min", default= 0.01, type= float)
    parser.add_argument("--epsilon-decrement", default= 0.001, type= float)
    parser.add_argument("--learning-rate", default= 0.0001, type= float)
    parser.add_argument("--batch-size", default= 128, type= int)
    parser.add_argument("--n-episodes", default= 100, type= int)
    parser.add_argument("--n-steps", default= 5000, type= int)
    parser.add_argument("--buffer-size", default= 1000000, type= int)
    parser.add_argument("--hid1-dim", default= 200, type= int)
    parser.add_argument("--hid2-dim", default= 128, type= int)
    parser.add_argument("--path", default= "", type= str)
    parser.add_argument("--tb-path", default= "", type= str)
    parser.add_argument("--displayEnv", default= False, type= bool)



    args = parser.parse_args()

# # prepare the visualisation window
# display = Display(visible=0, size=(1400, 900))
# display.start()

device = "cuda" if torch.cuda.is_available() else "cpu"

helper.fix_seed(args.seed)

env = gym.make('LunarLanderContinuous-v2')

agent1 = Agent(
    env=env, gamma=args.gamma, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decrement=args.epsilon_decrement, learning_rate=args.learning_rate, 
    batch_size=args.batch_size, n_episodes = args.n_episodes, n_steps = args.n_steps, buffer_size = args.buffer_size, hid1_dim=args.hid1_dim, hid2_dim=args.hid2_dim, 
    path=args.path, tb_path=args.tb_path, displayEnv=args.displayEnv)

agent1.train()

torch.save(agent1.target_net.state_dict(), args.path + "/agent.pt")

print("hey everything is done!")
