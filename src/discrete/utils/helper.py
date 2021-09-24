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





def show_video(directory):
  """
  Visualize the environments.
  ---
  INPUTS
    directory: (str) Ppath for the save directory.
  """
  html = []
  for mp4 in Path(directory).glob("*.mp4"):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

def display_env(env_name = 'LunarLander-v2', savePath = "./gym-results"):
  """
  Display one episode of gym environment.
  ---
  INPUTS
    env_name: (str) environment name as prescribed in gym library.
    savePath: (str) path to save the video.
  """
  # load env.
  env = gym.make(env_name)
  # wrap env in order to save our experiment on a file.
  env = Monitor(env, savePath, force=True, video_callable=lambda episode: True)

  done = False
  obs = env.reset()
  while not done:
      action = env.action_space.sample()
      obs, reward, done, info = env.step(action)
  env.close()
  show_video(savePath)
  
def fix_seed(seed):
  """
  Fix the ramdom seed.
  ---
  INPUTS
    seed: (int) the random seed.
  """
  random.seed(seed=seed)
  np.random.seed(seed=seed)
  torch.manual_seed(seed=seed)
  torch.cuda.manual_seed_all(seed=seed)