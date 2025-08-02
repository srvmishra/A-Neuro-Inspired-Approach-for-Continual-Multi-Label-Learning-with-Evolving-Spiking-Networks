import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, zero_one_loss, hamming_loss, jaccard_score
from sklearn.utils.multiclass import type_of_target
import torch
from random import shuffle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import markdown
import arff
import warnings
from scipy.io import loadmat
from continual_methods_new import *
from utils import *
from copy import deepcopy
warnings.filterwarnings("ignore")

## MODEL PARAMETERS
ARCHITECTURES = {'Yeast': [103, 13, 3, 10],
                'emotions': [72, 6, 3, 10],
                'flags': [19, 7, 2, 5],
                'gpositive': [440, 4, 3, 20],
                'gnegative': [440, 8, 3, 20],
                'plants': [440, 12, 3, 20],
                'virus': [440, 6, 3, 20],
                'birds': [260, 19, 3, 20],
                'Human': [440, 11, 3, 20],
                'Eukaryote': [440, 19, 3, 20],
                'scene': [294, 6, 3, 15],
                'delicious': [500, 983, 5, 50],
                'mediamill': [120, 101, 5, 50],
                'enron': [1001, 53, 5, 50],
                'FoodTruck': [21, 12, 2, 5]
                }
ckpt_dir = './checkpoints_new/'

## DATASET PARAMETERS
TRAINPATHS = {'Yeast': '../../datasets/yeast/yeast-train.arff',
              'emotions': '../../datasets/emotions/emotions-train.arff',
              'flags': '../../datasets/flags/flags-train.arff',
              'gpositive': '../../datasets/GpositivePseAAC/Gram_positivePseAAC519-train.mat',
              'gnegative': '../../datasets/GnegativePseAAC/Gram_negativePseAAC1392-train.mat',
              'plants': '../../datasets/PlantPseAAC/PlantPseAAC978-train.mat',
              'virus': '../../datasets/VirusPseAAC/VirusPseAAC207-train.mat',
              'birds': '../../datasets/birds/birds-train.arff',
              'Human': '../../datasets/HumanPseAAC/HumanPseAAC3106-train.mat',
              'Eukaryote': '../../datasets/EukaryotePseAAC/EukaryotePseAAC7766-train.mat',
              'scene': '../../datasets/scene/scene-train.arff',
              'delicious': '../../datasets/delicious/delicious-train.arff',
              'mediamill': '../../datasets/mediamill/mediamill-train.arff',
              'enron': '../../datasets/enron/enron-train.arff',
              'FoodTruck': '../../datasets/foodtruck/foodtruck-rand-hout-tra.arff',}
TESTPATHS = {'Yeast': '../../datasets/yeast/yeast-test.arff',
             'emotions': '../../datasets/emotions/emotions-test.arff',
             'flags': '../../datasets/flags/flags-test.arff',
             'gpositive': '../../datasets/GpositivePseAAC/Gram_positivePseAAC519-test.mat',
             'gnegative': '../../datasets/GnegativePseAAC/Gram_negativePseAAC1392-test.mat',
             'plants': '../../datasets/PlantPseAAC/PlantPseAAC978-test.mat',
             'virus': '../../datasets/VirusPseAAC/VirusPseAAC207-test.mat',
             'birds': '../../datasets/birds/birds-test.arff',
             'Human': '../../datasets/HumanPseAAC/HumanPseAAC3106-test.mat',
             'Eukaryote': '../../datasets/EukaryotePseAAC/EukaryotePseAAC7766-test.mat',
             'scene': '../../datasets/scene/scene-test.arff',
             'delicious': '../../datasets/delicious/delicious-test.arff',
             'mediamill': '../../datasets/mediamill/mediamill-test.arff',
             'enron': '../../datasets/enron/enron-test.arff',
             'FoodTruck': '../../datasets/foodtruck/foodtruck-rand-hout-tst.arff',}
NAMES = [n for n in ARCHITECTURES.keys()]
FEATURES = {n: k[0] for (n, k) in ARCHITECTURES.items()}

## CONTINUAL PARAMETERS
TASKS = {'Yeast': {3: {'samples': [500]*3, 'labels': [7] + [3]*2},
                   6: {'samples': [250]*6, 'labels': [2]*5 + [3]}},
         'emotions': {'samples': [130, 130, 131], 'labels': [2, 2, 2]},
         'flags': {'samples': [43, 43, 43], 'labels': [3, 2, 2]},
         'gpositive': {'samples': [103, 103, 105], 'labels': [2, 1, 1]},
         'gnegative': {'samples': [278, 280, 278], 'labels': [3, 2, 3]},
         'plants': {'samples': [186, 186, 186], 'labels': [4, 4, 4]},
         'virus': {'samples': [41, 41, 42], 'labels': [2, 2, 2]},
         'birds': {'samples': [54]*5 + [52], 'labels': [3]*5 + [4]},
         'Human': {'samples': [310]*4 + [622], 'labels': [2]*4 + [3]},
         'Eukaryote': {'samples': [435, 530, 438] + [465]*7, 'labels': [2]*9 + [1]},
         'scene': {'samples': [405, 403, 403], 'labels': [2, 2, 2]},
         'delicious': {20: {'samples': [646]*20, 'labels': [49]*19 + [52]}, 
                       50: {'samples': [258]*49 + [278], 'labels':[20]*49 + [3]}},
         'mediamill': {20: {'samples': [1549]*19 + [1562], 'labels': [5]*19 + [6]}, 
                       50: {'samples': [619]*49 + [662], 'labels': [2]*49 + [3]}},
         'enron': {'samples': [112]*9 + [115], 'labels': [5]*9 + [8]},
         'FoodTruck': {'samples': [90, 80, 80], 'labels': [4, 4, 4]}}

# debug
# run
# 30 epochs

class Hyperparameters(object):
    def __init__(self, dataset_name, continual_method, device, batch_size=8, seed=0, lr=0.1, epochs=1, 
                 loss_type='asy', measure='imb. av. f1', num_tasks=None):
        self.name = dataset_name
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.loss_type = loss_type
        self.ckpt_dir = ckpt_dir
        self.measure = measure
        self.num_tasks = num_tasks

        # model parameters
        self.num_inputs, self.num_outputs, self.num_hidden_layers, self.num_neurons_per_layer = ARCHITECTURES[self.name]
        
        # dataset parameters
        self.train_path = TRAINPATHS[self.name]
        self.test_path = TESTPATHS[self.name]
        self.num_features = FEATURES[self.name]

        self.generator = np.random.RandomState(self.seed)
        self.select_loss()
        self.select_task_list()

        # continual learning baseline parameters
        self.mode = continual_method
        self.continual_params = {'importance': 0.1}

    def select_loss(self):
        if self.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        if self.loss_type == 'asy':
            self.criterion = AsymmetricLossOptimized()
        if self.loss_type == 'corr':
            self.criterion = CorrelationLoss(self.device)
        if self.loss_type == 'both':
            self.criterion = CorrelationAsymmetricLoss(self.device)  

    def select_task_list(self):
        self.tasks = TASKS[self.name]
        if self.name in ['delicious', 'mediamill'] and self.num_tasks in [20, 50]:
            self.tasks = self.tasks[self.num_tasks]
        elif self.name == 'Yeast' and self.num_tasks in [3, 6]:
            self.tasks = self.tasks[self.num_tasks]
        self.num_tasks = len(self.tasks['samples'])