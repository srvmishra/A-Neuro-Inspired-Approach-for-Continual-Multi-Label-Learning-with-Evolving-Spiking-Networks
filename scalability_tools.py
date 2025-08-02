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
        self.n_inputs = hparams['inputs']
        self.n_outputs = hparams['outputs']
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
    
    def neuron_addition_operation(self, train_spikes, train_labels, verbose=False):
        start_time = time.time()
        for j in range(len(train_spikes)):
            # x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
            x = train_spikes[j].view(1, -1)
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
        end_time = time.time()
        elapsed_time = end_time - start_time

        class1_neurons, class0_neurons = 0, 0
        for i, net in enumerate(self.nets):
            class1_neurons = class1_neurons + len([i for i in range(len(net.nature)) if net.nature[i] == 1])
            class0_neurons = class0_neurons + len([i for i in range(len(net.nature)) if net.nature[i] == 0])
            # print("Class {}, class 1 neurons = {}, class 0 neurons = {}".format(i, len(c1), len(c0)))
        
        if verbose:
            return elapsed_time, class1_neurons, class0_neurons
    
    def fit(self, train_spikes, train_labels, verbose=False):
        indices = np.array([k for k in range(len(train_spikes))])
        
        if not self.retrain:
            self.neuron_addition_operation(train_spikes, train_labels)
            # train_precs, train_recs, train_f1s, train_accs = [], [], [], []
            # test_precs, test_recs, test_f1s, test_accs = [], [], [], []
            start, stop = 0, self.epochs
            self.results = {}
        else:
            start, stop = self.results['epochs'], self.results['epochs'] + self.results['epochs_left']
            print("Retraining from {} mode".format(self.results['mode']))
        
        start_time = time.time()
        for k in tqdm(range(start, stop)): 
            self.generator.shuffle(indices)
            for j in indices:
                # x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
                x = train_spikes[j].view(1, -1)
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
        end_time = time.time()
        elapsed_time = end_time - start_time

        if verbose:
            return elapsed_time
            

def neuron_addition_samples(data, labels, device, num_features, num_labels, num_samples_list, hparams):
    data_ = data[:, :num_features]

    a = data_.max(axis=0)
    b = data_.min(axis=0)

    data_spikes = encoding(data_, device, a, b)
    data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)
    labels_ = labels[:, :num_labels]

    hparams['inputs'] = num_features * 6
    hparams['outputs'] = num_labels

    times, class1, class0 = [], [], []

    for num_samples in num_samples_list:
        data_spikes_ = data_spikes[:num_samples]
        labels_new = labels_[:num_samples, :]
        net = OverallSNN(hparams)
        elapsed_time, class1_neurons, class0_neurons = net.neuron_addition_operation(data_spikes_, labels_new, verbose=True)
        times.append(elapsed_time)
        class1.append(class1_neurons)
        class0.append(class0_neurons)

    return times, class1, class0

def weight_update_samples(data, labels, device, num_features, num_labels, num_samples_list, hparams):
    data_ = data[:, :num_features]

    a = data_.max(axis=0)
    b = data_.min(axis=0)

    data_spikes = encoding(data_, device, a, b)
    data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)
    labels_ = labels[:, :num_labels]

    hparams['inputs'] = num_features * 6
    hparams['outputs'] = num_labels

    times = []

    for num_samples in num_samples_list:
        data_spikes_ = data_spikes[:num_samples]
        labels_new = labels_[:num_samples, :]
        net = OverallSNN(hparams)
        elapsed_time = net.fit(data_spikes_, labels_new, verbose=True)
        times.append(elapsed_time/hparams['n_epochs'])

    return times

def neuron_addition_features(data, labels, device, num_features_list, num_labels, num_samples, hparams):
    labels_ = labels[:num_samples, :num_labels]

    hparams['outputs'] = num_labels

    times, class1, class0 = [], [], []

    for num_features in num_features_list:
        data_ = data[:num_samples, :num_features]
        a = data_.max(axis=0)
        b = data_.min(axis=0)

        data_spikes = encoding(data_, device, a, b)
        data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)

        hparams['inputs'] = num_features * 6

        net = OverallSNN(hparams)
        elapsed_time, class1_neurons, class0_neurons = net.neuron_addition_operation(data_spikes, labels_, verbose=True)
        times.append(elapsed_time)
        class1.append(class1_neurons)
        class0.append(class0_neurons)

    return times, class1, class0

def weight_update_features(data, labels, device, num_features_list, num_labels, num_samples, hparams):
    labels_ = labels[:num_samples, :num_labels]

    hparams['outputs'] = num_labels

    times = []

    for num_features in num_features_list:
        data_ = data[:num_samples, :num_features]
        a = data_.max(axis=0)
        b = data_.min(axis=0)

        data_spikes = encoding(data_, device, a, b)
        data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)

        hparams['inputs'] = num_features * 6

        net = OverallSNN(hparams)
        elapsed_time = net.fit(data_spikes, labels_, verbose=True)
        times.append(elapsed_time/hparams['n_epochs'])

    return times

def neuron_addition_labels(data, labels, device, num_features, num_labels_list, num_samples, hparams):
    data_ = data[:num_samples, :num_features]
    labels_ = labels[:num_samples, :]

    a = data_.max(axis=0)
    b = data_.min(axis=0)

    data_spikes = encoding(data_, device, a, b)
    data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)
    
    hparams['inputs'] = num_features * 6
    
    times, class1, class0 = [], [], []

    for num_labels in num_labels_list:
        labels_new = labels_[:, :num_labels]
        hparams['outputs'] = num_labels
        net = OverallSNN(hparams)
        # print(len(net.nets))
        # print()
        elapsed_time, class1_neurons, class0_neurons = net.neuron_addition_operation(data_spikes, labels_new, verbose=True)
        times.append(elapsed_time)
        class1.append(class1_neurons)
        class0.append(class0_neurons)

    return times, class1, class0

def weight_update_labels(data, labels, device, num_features, num_labels_list, num_samples, hparams):
    data_ = data[:num_samples, :num_features]
    labels_ = labels[:num_samples, :]

    a = data_.max(axis=0)
    b = data_.min(axis=0)

    data_spikes = encoding(data_, device, a, b)
    data_spikes = torch.from_numpy(np.array(data_spikes)).to(device)
    
    hparams['inputs'] = num_features * 6
    
    times = []

    for num_labels in num_labels_list:
        labels_new = labels_[:, :num_labels]
        hparams['outputs'] = num_labels
        net = OverallSNN(hparams)
        elapsed_time = net.fit(data_spikes, labels_new, verbose=True)
        times.append(elapsed_time/hparams['n_epochs'])

    return times