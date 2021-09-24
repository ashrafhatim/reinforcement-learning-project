import os
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd

import seaborn as sns
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output
from pathlib import Path

import random, os.path, math, glob, csv, base64, itertools, sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import io
from IPython.display import HTML

from torch.utils.tensorboard import SummaryWriter

from pyvirtualdisplay import Display

from models import Agent as Agent
from models import DQN as DQN
from models import ReplayBuffer as ReplayBuffer
from utils import helper as helper

import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser()
      
    parser.add_argument("--env-name", default= "LunarLander-v2", type= str)
    parser.add_argument("--gamma", default= 0.99, type= float)
    parser.add_argument("--epsilon", default= 1, type= float)
    parser.add_argument("--epsilon-min", default= 0.01, type= float)
    parser.add_argument("--epsilon-decrement", default= 0.001, type= float)
    parser.add_argument("--learning-rate", default= 0.0001, type= float)
    parser.add_argument("--batch-size", default= 128, type= int)
    parser.add_argument("--n-episodes", default= 700, type= int)
    parser.add_argument("--n-steps", default= 5000, type= int)
    parser.add_argument("--buffer-size", default= 1000000, type= int)
    parser.add_argument("--hid1-dim", default= 200, type= int)
    parser.add_argument("--hid2-dim", default= 128, type= int)
    parser.add_argument("--path", default= "", type= str)
    parser.add_argument("--tb-path", default= "", type= str)
    parser.add_argument("--printLog", default= False, type= bool)


    args = parser.parse_args()

# prepare the visualisation window
display = Display(visible=0, size=(1400, 900))
display.start()

device = "cuda" if torch.cuda.is_available() else "cpu"

helper.fix_seed(0)


agent1 = Agent(
    env_name=args.env_name, gamma=args.gamma, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decrement=args.epsilon_decrement, learning_rate=args.learning_rate, 
    batch_size=args.batch_size, n_episodes = args.n_episodes, n_steps = args.n_steps, buffer_size = args.buffer_size, hid1_dim=args.hid1_dim, hid2_dim=args.hid2_dim, 
    path=args.path, tb_path=args.tb_path, device = device, printLog = args.printLog)

agent1.train()


# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [ transforms.Resize([28,28]), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         ),
#     ),
#     batch_size=64,
#     shuffle=True,
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# exp_num = args.exp_num 
# epochs = args.epochs
# TENSORBOARD_PATH = args.TENSORBOARD-PATH # dataset folder
# SAVE_PATH = args.SAVE-PATH # save folder


# generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler = get_model.get_model(1, 1, 100, (1,28,28), lr = 5e-5)
# train.train(generator, discriminator, optimizer_G, optimizer_D, optimizer_G_scheduler, optimizer_D_scheduler , epochs=epochs, dataloader=dataloader, save_path = SAVE_PATH, tensorboard_path = TENSORBOARD_PATH)

# helper.sample_img(generator)