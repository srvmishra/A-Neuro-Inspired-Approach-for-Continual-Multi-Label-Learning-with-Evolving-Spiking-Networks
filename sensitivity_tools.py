import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, zero_one_loss, hamming_loss, jaccard_score
import torch
from random import shuffle
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import markdown
from sklearn.feature_selection import RFECV
from sklearn.utils.multiclass import type_of_target
from MultiLabelEvolvingSNN import encoding, SingleSNN

class OverallSNN:
    def __init__(self, hparams):
        self.name = hparams['name']      # field added later to facilitate saving of graphs
        self.n_inputs = 12
        self.n_outputs = 1
        self.tau = hparams['time_constant']
        self.alpha_a = hparams['alpha_a']
        self.weights_max_limit = hparams['w_max']
        self.weights_min_limit = hparams['w_min']
        self.alpha_m = hparams['alpha_m']
        self.epochs = hparams['n_epochs']
        self.print_every = hparams['print_every']
        self.lr = hparams['lr']
        self.third_time = int(hparams['sim_time']/3)
        self.tm = self.alpha_m*2*self.third_time
        self.ta = self.alpha_a*hparams['sim_time']
        self.seed = hparams['seed']
        self.device = hparams['device']
        self.ckpt_dir = hparams['ckpt_dir']
        self.load_from_file = hparams['load_from_file']
        self.times = torch.linspace(0, hparams['sim_time'], hparams['sim_time']+1).unsqueeze(0).to(self.device)
        self.generator = np.random.RandomState(self.seed)
        self.format = '{}_{}'.format(self.lr, self.epochs)
#         self.hparams = hparams
        if self.load_from_file is not None:
            self.init_weights(filepath=self.load_from_file)
            for net in self.nets:
                net.device = self.device
                net.weights = net.weights.to(self.device)
                net.theta = net.theta.to(self.device)
                net.sim_time = net.sim_time.to(self.device)
            self.retrain = True
        else:
            self.init_weights()
            self.retrain = False
        
    def init_weights(self, filepath=None):
        if filepath is not None:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.nets, self.results = data[0], data[1]
            f.close()
        else:
            self.nets = [SingleSNN(self.third_time, self.ta, self.tm, self.device, self.weights_max_limit, self.weights_min_limit, self.lr, self.times) for i in range(self.n_outputs)]
        
    def srf(self, s):
        f = s*torch.exp(1.0 - s/self.tau)/self.tau
        f[torch.where(s<=0)] = 0.0
        return f.double()
    
    def neuron_addition_operation(self, train_spikes, train_labels):
        for j in range(len(train_spikes)):
            x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            for i, net in enumerate(self.nets):
                y = train_labels[j][i]
                if y == 1:
                    nature = 'cc'
                    cc = 1
                    oc = 0
                else:
                    nature = 'oc'
                    cc = 0
                    oc = 1
                net.add_neuron(srf, cc)
        for i, net in enumerate(self.nets):
            # because we are only considering one label in sensitivity experiments
            # otherwise we need to set net.c1 and net.c0 to these values
            self.c1 = len([i for i in range(len(net.nature)) if net.nature[i] == 1])
            self.c0 = len([i for i in range(len(net.nature)) if net.nature[i] == 0])
            # print("Class {}, class 1 neurons = {}, class 0 neurons = {}".format(i, len(c1), len(c0)))
        # return c1, c0
    
    def fit(self, train_spikes, train_labels, test_spikes, test_labels):
        indices = np.array([k for k in range(len(train_spikes))])
        if not self.retrain:
            self.neuron_addition_operation(train_spikes, train_labels)
            start, stop = 0, self.epochs
            self.results = {}
        else:
            start, stop = self.results['epochs'], self.results['epochs'] + self.results['epochs_left']
            print("Retraining from {} mode".format(self.results['mode']))
        for k in tqdm(range(start, stop)): 
            self.generator.shuffle(indices)
            for j in indices:
                x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
                s = x.T - self.times
                srf = self.srf(-1.0*s)
                for i, net in enumerate(self.nets):
                    y = train_labels[j][i] 
                    if y == 1:
                        nature = 'cc'
                        cc = 1
                        oc = 0
                    else:
                        nature = 'oc'
                        cc = 0
                        oc = 1
                    net.update(srf, cc)
            
        accuracy, avg_spk_diff = self.evaluate(test_spikes, test_labels)
        return accuracy, avg_spk_diff
    
    def evaluate(self, data, labels):
        pred_labels = []
        true_labels = []
        for j in range(len(data)):
            x = torch.from_numpy(data[j]).view(1, -1).to(self.device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            pred_label = []
            spike_diffs = []
            true_labels.append(labels[j])
            for i, net in enumerate(self.nets):
                net.forward(srf)
                pred = 0
                if net.class1_spikes < net.class0_spikes:
                    pred = 1
                # spike_diffs.append(np.abs(net.class1_spikes - net.class0_spikes))
                # just the average, no absolute value
                spike_diffs.append(net.class1_spikes - net.class0_spikes)
                pred_label.append(pred)
                net.clear_buffers()
            pred_labels.append(pred_label[:self.n_outputs])
        avg_spk_diff = sum(spike_diffs)/len(spike_diffs)
        true_labels = np.array(true_labels)[:, :self.n_outputs]
        pred_labels = np.array(pred_labels)
#         print(true_labels, pred_labels)

        # num_zeros = (true_labels == 0).sum(0)
        # num_ones = (true_labels == 1).sum(0)

        # if num_zeros > num_ones:
        #     pos_correct = np.where((true_labels == 1) & (pred_labels == 1))[0]
        #     n_correct = pred_labels[pos_correct].sum()
        #     class_accs = n_correct/num_ones
        # elif num_zeros < num_ones:
        #     pos_correct = np.where((true_labels == 0) & (pred_labels == 0))[0]
        #     n_correct = pred_labels[pos_correct].sum()
        #     class_accs = n_correct/num_zeros
        # else:
        #     class_accs = (true_labels==pred_labels).sum(0)/len(data)

        class_accs = (true_labels==pred_labels).sum(0)/len(data)
        return class_accs, avg_spk_diff
       

