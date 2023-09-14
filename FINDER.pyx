#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division
import torch
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
