# -*- coding: utf-8 -*-
import torch

from FINDER import FINDERconv, LossFn

import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph                # C++
import graph                            # C++
import nstep_replay_mem                 # C++
import nstep_replay_mem_prioritized     # C++
import mvc_env                          # C++
import utils                            # C++
import scipy.linalg as linalg
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train():
    model = FINDERconv()
    model.train()

    criterion = LossFn()




if __name__=="__main__":
    pass
