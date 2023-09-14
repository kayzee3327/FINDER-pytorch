#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
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

# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 500000
cdef double LEARNING_RATE = 0.0001   #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30
cdef int NUM_MAX = 50
cdef int REG_HIDDEN = 32
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.01  # 权重初始化的方差
cdef int n_valid = 200
cdef int aux_dim = 4
cdef int num_env = 1
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage

class FINDER(nn.Module):
    def __init__(self):
        super().__init__()

        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'barabasi_albert'  #erdos_renyi, powerlaw, small-world， barabasi_albert
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsMultiStepDQN = True  ##(if IsNStepDQN=False, N_STEP==1)
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list = []
        self.g_list = []
        self.pred = []
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon, alpha, beta,
                                                                         beta_increment_per_sampling, TD_err_upper,
                                                                         MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)

