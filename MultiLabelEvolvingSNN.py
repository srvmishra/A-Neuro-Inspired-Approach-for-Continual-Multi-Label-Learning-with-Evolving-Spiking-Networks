import os
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

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

def encoding(data, device, a, b):
    # change according to dataset
    num_gaussians = 6
    num_features = data.shape[1]
#     print(data.shape)
    means = torch.Tensor([(2*h-3)/(2*(num_gaussians-2)) for h in range(1, num_gaussians+1)]).to(device).float()
    std_dev = 1.0/(0.7*(num_gaussians-2))
    normalize = lambda x, a, b: (x - b)/(a - b)
#     a = data[:, :num_features].max(axis=0)
#     b = data[:, :num_features].min(axis=0)
    data[:, :num_features] = normalize(data[:, :num_features], a, b)
    T = 300
    spiked_matrices = []
#     spiked_labels = []
    # for i in tqdm(range(len(data))):
    for i in range(len(data)):
        x = torch.from_numpy(data[i, :num_features]).to(device).view(-1, 1).float()
        spike = (1.0 - torch.exp(-(x-means)**2/(2*std_dev**2)))*T
        spike = spike.view(1, -1).cpu().numpy()
        assert spike.shape == (1, num_features*num_gaussians)
        spiked_matrices.append(spike)
#         spiked_labels.append(data[i, num_features:])
    return spiked_matrices

def encode_data_to_spikes(data, device, num_features, a, b):
    # change according to dataset
    num_gaussians = 6
    means = torch.Tensor([(2*h-3)/(2*(num_gaussians-2)) for h in range(1, num_gaussians+1)]).to(device).float()
    std_dev = 1.0/(0.7*(num_gaussians-2))
    normalize = lambda x, a, b: (x - b)/(a - b)
#     a = data[:, :num_features].max(axis=0)
#     b = data[:, :num_features].min(axis=0)
    data[:, :num_features] = normalize(data[:, :num_features], a, b)
    T = 300
    spiked_matrices = []
    spiked_labels = []
    for i in tqdm(range(len(data))):
        x = torch.from_numpy(data[i, :num_features]).to(device).view(-1, 1).float()
        spike = (1.0 - torch.exp(-(x-means)**2/(2*std_dev**2)))*T
        spike = spike.view(1, -1).cpu().numpy()
        assert spike.shape == (1, num_features*num_gaussians)
        spiked_matrices.append(spike)
        spiked_labels.append(data[i, num_features:])
    return spiked_matrices, spiked_labels

def one_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    err = 0
    for i in range(y_true.shape[0]):
        err = err + (y_true[i, :] != y_pred[i, :]).all()
    val = err/y_true.shape[0]
    return val

class SingleSNN:
    def __init__(self, target_time, ta, tm, device, w_max, w_min, lr, sim_time):
        self.third_time = target_time
        self.device = device
        self.weights_min_limit, self.weights_max_limit = w_max, w_min
        self.lr = lr
        self.sim_time = sim_time
        self.ta = ta
        self.tm = tm
        self.init_weights()
    
    def init_weights(self):
        self.weights = []
        self.theta = None
        self.nature = []
    
    def compute_times(self, v, t, locs):
        spikes = []
        for i, p in enumerate(v):
            a = torch.where(p > t[i])
            if len(a[0]) == 0:
                spikes.append(int(self.sim_time[0, -1].item()))
            else:
                spikes.append(self.sim_time[0, torch.min(a[0])].item())
        spikes = torch.tensor(spikes).to(self.device)
        if torch.min(spikes) == self.sim_time[0, -1]:
            return self.sim_time[0, -1].item(), locs[torch.argmin(spikes).item()]
        else:
            return torch.min(spikes).item(), locs[torch.argmin(spikes).item()]
    
    def add_neuron(self, srf, cc):
        v = srf[:, self.third_time].view(-1, 1).to(self.device)
        norm_srf = srf/srf.sum(0)
        u = norm_srf[:, self.third_time].view(1, -1).to(self.device)
#         print(self.nature)
        if len(self.nature) > 0:
            class1_pos = [i for i in range(len(self.nature)) if self.nature[i] == 1]
            class0_pos = [i for i in range(len(self.nature)) if self.nature[i] == 0]
            class1_weights = self.weights[class1_pos, :]
            class0_weights = self.weights[class0_pos, :]
            
#             print(class1_pos)
#             print(class0_pos)
        if cc == 1:
            if len(self.weights) > 0:
                if len(class1_weights) > 0:
                    self.class1_spikes, _, _, _ = self.forward(srf)
                    if self.class1_spikes > self.ta:
                        self.weights = torch.cat([self.weights, u], dim=0)
                        self.theta = torch.cat([self.theta, torch.matmul(u, v)], dim=0)
                        self.nature.append(cc)
                    if len([i for i in range(len(self.nature)) if self.nature[i] == 0]) > 0:
                        self.update(srf, 1)
                else:
                    self.weights = torch.cat([self.weights, u], dim=0)
                    self.theta = torch.cat([self.theta, torch.matmul(u, v)], dim=0)
                    self.nature.append(cc)
            else:
                self.weights = u
                self.theta = torch.matmul(u, v).view(-1, 1) 
                self.nature.append(cc)
        else:
            if len(self.weights) > 0:
                if len(class0_weights) > 0:
                    _, self.class0_spikes, _, _ = self.forward(srf)
                    if self.class0_spikes > self.ta:
                        self.weights = torch.cat([self.weights, u], dim=0)
                        self.theta = torch.cat([self.theta, torch.matmul(u, v)], dim=0)
                        self.nature.append(cc)
                    if len([i for i in range(len(self.nature)) if self.nature[i] == 1]) > 0:
                        self.update(srf, 0)
                else:
                    self.weights = torch.cat([self.weights, u], dim=0)
                    self.theta = torch.cat([self.theta, torch.matmul(u, v)], dim=0)
                    self.nature.append(cc)
            else:
                self.weights = u
                self.theta = torch.matmul(u, v).view(-1, 1)   
                self.nature.append(cc)
    
    def forward(self, srf):
        class1_pos = [i for i in range(len(self.nature)) if self.nature[i] == 1]
        class0_pos = [i for i in range(len(self.nature)) if self.nature[i] == 0]
#         print(class1_pos)
#         print(class0_pos)
#         print(self.weights.shape)
        class1_weights = self.weights[class1_pos, :]
        class0_weights = self.weights[class0_pos, :]
        class1_theta = self.theta[class1_pos, :]
        class0_theta = self.theta[class0_pos, :]
#         print(self.nature)
#         print(class0_pos)
#         print(class1_pos)
#         print(class0_pos)
        
        if len(class1_weights) > 0:
            self.v1 = torch.matmul(class1_weights, srf)
            self.class1_spikes, self.class1_locs = self.find_spikes(class1_theta, class1_pos, '1')
        if len(class0_weights) > 0:
            self.v0 = torch.matmul(class0_weights, srf)
            self.class0_spikes, self.class0_locs = self.find_spikes(class0_theta, class0_pos, '0')
        if len(class1_weights) == 0:
            self.class1_spikes, self.class1_locs = None, None
        if len(class0_weights) == 0:
            self.class0_spikes, self.class0_locs = None, None
        return self.class1_spikes, self.class0_spikes, self.class1_locs, self.class0_locs
    
    def find_spikes(self, theta, locs, nature):
#         print(nature)
        if nature == '1':
            time_1, loc_1 = self.compute_times(self.v1, theta, locs)
            spikes = time_1
            locs = loc_1
            return spikes, locs
        if nature == '0':
            time_0, loc_0 = self.compute_times(self.v0, theta, locs) 
            spikes = time_0
            locs = loc_0
            return spikes, locs
    
    def predict(self, srf):
        self.forward(srf)
        pred = 0
        if self.class1_spikes < self.class0_spikes:
            pred = 1
        self.clear_buffers()
        return pred
    
    def update(self, srf, cc):
#         class1_pos = [i for i in range(len(self.nature)) if self.nature[i] == 1]
#         class0_pos = [i for i in range(len(self.nature)) if self.nature[i] == 0]
#         print(class1_pos)
#         print(class0_pos)
#         print(self.weights.shape)
#         class1_weights = self.weights[class1_pos, :]
#         class0_weights = self.weights[class0_pos, :]
#         class1_theta = self.theta[class1_pos, :]
#         class0_theta = self.theta[class0_pos, :]
        self.clear_buffers()
        self.forward(srf)
#         print(self.class1_spikes, self.class0_spikes)
        u = srf/srf.sum(0)
        self.weights_update = torch.zeros_like(self.weights).to(self.device)
        
        if cc == 1:
            cc_spikes = self.class1_spikes
            cc_locs = self.class1_locs
            oc_spikes = self.class0_spikes
            oc_locs = self.class0_locs
        else:
            oc_spikes = self.class1_spikes
            oc_locs = self.class1_locs
            cc_spikes = self.class0_spikes
            cc_locs = self.class0_locs
        
        if oc_spikes - cc_spikes < self.tm:
            dw_cc = u[:, int(cc_spikes)]
            dw_oc = u[:, int(oc_spikes)]
            self.weights_update[int(cc_locs), :] = dw_cc
            self.weights_update[int(oc_locs), :] = dw_oc
            self.weights_update = torch.nan_to_num(self.weights_update)
            indices = torch.where(self.weights_update > self.weights)
            self.weights_update[int(oc_locs), :] = -1.0*dw_oc
            self.weights[indices] += self.lr*self.weights_update[indices]
#             self.weights = torch.clamp(self.weights, self.weights_min_limit, self.weights_max_limit)
            
    def clear_buffers(self):
        self.weights_update = None
        self.v1 = None
        self.v0 = None
        self.class0_spikes = None
        self.class1_spikes = None
        self.class1_locs = None
        self.class0_locs = None
    
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
            c1 = [i for i in range(len(net.nature)) if net.nature[i] == 1]
            c0 = [i for i in range(len(net.nature)) if net.nature[i] == 0]
            print("Class {}, class 1 neurons = {}, class 0 neurons = {}".format(i, len(c1), len(c0)))
    
    def fit(self, train_spikes, train_labels, test_spikes, test_labels):
        indices = np.array([k for k in range(len(train_spikes))])
        if not self.retrain:
            self.neuron_addition_operation(train_spikes, train_labels)
            train_precs, train_recs, train_f1s, train_accs = [], [], [], []
            test_precs, test_recs, test_f1s, test_accs = [], [], [], []
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
            
            prec_train, rec_train, f1_train, train_class_accs = self.evaluate(train_spikes, train_labels)
            train_precs.append(prec_train)
            train_recs.append(rec_train)
            train_f1s.append(f1_train)
            train_accs.append(train_class_accs)
            prec_test, rec_test, f1_test, test_class_accs = self.evaluate(test_spikes, test_labels)
            test_precs.append(prec_test)
            test_recs.append(rec_test)
            test_f1s.append(f1_test)
            test_accs.append(test_class_accs)
            
            if prec_test == max(test_precs):
                self.results['mode'] = 'Best Test Precision'
                self.results['epochs'] = k + 1
                self.results['epochs_leftover'] = self.epochs - self.results['epochs']
                self.results['max_test_prec'] = prec_test
                self.results['test_rec_at_max_test_prec'] = rec_test
                self.results['test_f1_at_max_test_prec'] = f1_test
                self.results['train_prec_at_max_test_prec'] = prec_train
                self.results['train_rec_at_max_test_prec'] = rec_train
                self.results['train_f1_at_max_test_prec'] = f1_train
                self.results['train_classwise_accuracies'] = train_accs
                self.results['test_classwise_accuracies'] = test_accs
#                 self.results['train_precs'] = train_precs
#                 self.results['train_recs'] = train_recs
#                 self.results['train_f1s'] = train_f1s
#                 self.results['test_precs'] = test_precs
#                 self.results['test_recs'] = test_recs
#                 self.results['test_f1s'] = test_f1s
                self.save('best_prec_model' + self.format)
                
            if rec_test == max(test_recs):
                self.results['mode'] = 'Best Test Recall'
                self.results['epochs'] = k + 1
                self.results['epochs_leftover'] = self.epochs - self.results['epochs']
                self.results['max_test_rec'] = rec_test
                self.results['test_prec_at_max_test_rec'] = prec_test
                self.results['test_f1_at_max_test_rec'] = f1_test
                self.results['train_prec_at_max_test_rec'] = prec_train
                self.results['train_rec_at_max_test_rec'] = rec_train
                self.results['train_f1_at_max_test_rec'] = f1_train
                self.results['train_classwise_accuracies'] = train_accs
                self.results['test_classwise_accuracies'] = test_accs
#                 self.results['train_precs'] = train_precs
#                 self.results['train_recs'] = train_recs
#                 self.results['train_f1s'] = train_f1s
#                 self.results['test_precs'] = test_precs
#                 self.results['test_recs'] = test_recs
#                 self.results['test_f1s'] = test_f1s
                self.save('best_rec_model' + self.format)
                
            if f1_test == max(test_f1s):
                self.results['mode'] = 'Best Test F1'
                self.results['epochs'] = k + 1
                self.results['epochs_leftover'] = self.epochs - self.results['epochs']
                self.results['max_test_f1'] = f1_test
                self.results['test_prec_at_max_test_f1'] = prec_test
                self.results['test_rec_at_max_test_f1'] = rec_test
                self.results['train_prec_at_max_test_f1'] = prec_train
                self.results['train_rec_at_max_test_f1'] = rec_train
                self.results['train_f1_at_max_test_f1'] = f1_train
                self.results['train_classwise_accuracies'] = train_accs
                self.results['test_classwise_accuracies'] = test_accs
#                 self.results['train_precs'] = train_precs
#                 self.results['train_recs'] = train_recs
#                 self.results['train_f1s'] = train_f1s
#                 self.results['test_precs'] = test_precs
#                 self.results['test_recs'] = test_recs
#                 self.results['test_f1s'] = test_f1s
                self.save('best_f1_model' + self.format)
            
            if (k+1)%self.print_every == 0:                
                print("Epoch number: {}".format(k+1))
                print("Train Precision: {:.4f}, Test Precision: {:.4f}".format(prec_train, prec_test))
                print("Train Recall: {:.4f}, Test Recall: {:.4f}".format(rec_train, rec_test))
                print("Train F1: {:.4f}, Test F1: {:.4f}".format(f1_train, f1_test))
                print("Max. train classwise accuracy for is {:.4f}".format(max(train_class_accs)))
                print("Min. train classwise accuracy for is {:.4f}".format(min(train_class_accs)))
                print("Max. test classwise accuracy for is {:.4f}".format(max(test_class_accs)))
                print("Min. test classwise accuracy for is {:.4f}".format(min(test_class_accs)))
                
        train_history = (train_precs, train_recs, train_f1s)
        test_history = (test_precs, test_recs, test_f1s)
        
        self.results['train_history'] = train_history
        self.results['test_history'] = test_history
        
        index = test_precs.index(max(test_precs))
        print("Max. Test Precision: {:.4f} at epoch {}".format(max(test_precs), index+1))
        print("Test Recall at max. test precision: {:.4f}".format(test_recs[index]))
        print("Test F1 at max. test precision: {:.4f}".format(test_f1s[index]))
        print("Train Precision at max. test precision: {:.4f}".format(train_precs[index]))
        print("Train Recall at max. test precision: {:.4f}".format(train_recs[index]))
        print("Train F1 at max. test precision: {:.4f}".format(train_f1s[index]))
        
        index = test_recs.index(max(test_recs))
        print("Max. Test Recall: {:.4f} at epoch {}".format(max(test_recs), index+1))
        print("Test Precision at max. test recall: {:.4f}".format(test_precs[index]))
        print("Test F1 at max. test recall: {:.4f}".format(test_f1s[index]))
        print("Train Recall at max. test recall: {:.4f}".format(train_recs[index]))
        print("Train Precision at max. test recall: {:.4f}".format(train_precs[index]))
        print("Train F1 at max. test recall: {:.4f}".format(train_f1s[index]))
        
        index = test_f1s.index(max(test_f1s))
        print("Max. Test F1: {:.4f} at epoch {}".format(max(test_f1s), index+1))
        print("Test Recall at max. test F1: {:.4f}".format(test_recs[index]))
        print("Test Precision at max. test F1: {:.4f}".format(test_precs[index]))
        print("Train Recall at max. test F1: {:.4f}".format(train_recs[index]))
        print("Train Precision at max. test F1: {:.4f}".format(train_precs[index]))
        print("Train F1 at max. test F1: {:.4f}".format(train_f1s[index]))
        
        self.plot_results()
        
        return train_history, test_history
    
    def evaluate(self, data, labels):
        pred_labels = []
        true_labels = []
        for j in range(len(data)):
            x = torch.from_numpy(data[j]).view(1, -1).to(self.device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            pred_label = []
            true_labels.append(labels[j])
            for i, net in enumerate(self.nets):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label[:self.n_outputs])
        
        true_labels = np.array(true_labels)[:, :self.n_outputs]
        pred_labels = np.array(pred_labels)
#         print(true_labels, pred_labels)
        class_accs = (true_labels==pred_labels).sum(0)/len(data)
        prec = precision_score(true_labels, pred_labels, average='micro')
        rec = recall_score(true_labels, pred_labels, average='micro')
        f1 = f1_score(true_labels, pred_labels, average='micro')
        return prec, rec, f1, class_accs
    
    def save(self, name):
        filepath = os.path.join(self.ckpt_dir, name + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump([self.nets, self.results], f)
        f.close()
#         print("File saved to {}".format(filepath))
        return
    
    def plot_results(self):
#         train_precs, train_recs, train_f1s = self.results['train_precs'], self.results['train_recs'], self.results['train_f1s']
#         test_precs, test_recs, test_f1s = self.results['test_precs'], self.results['test_recs'], self.results['test_f1s']
        
        if 'train_history' in self.results.keys() and 'test_history' in self.results.keys():
            train_precs, train_recs, train_f1s = self.results['train_history']
            test_precs, test_recs, test_f1s = self.results['test_history']
        elif 'train_precs' in self.results.keys() and 'test_precs' in self.results.keys():
            train_precs, train_recs, train_f1s = self.results['train_precs'], self.results['train_recs'], self.results['train_f1s']
            test_precs, test_recs, test_f1s = self.results['test_precs'], self.results['test_recs'], self.results['test_f1s']
        else:
            return

        plt.figure(figsize=(10, 10))
        plt.plot(train_precs, label='train')
        plt.plot(test_precs, label='test')
        plt.legend(fontsize=25)
        plt.xlabel('epochs', fontsize=25)
        plt.ylabel('precision', fontsize=25)
        plt.title('Precisions', fontsize=25)
        filepath = os.path.join('./figures', self.name + '_prec' + '.png')
        plt.savefig(filepath)
        
        plt.figure(figsize=(10, 10))
        plt.plot(train_recs, label='train')
        plt.plot(test_recs, label='test')
        plt.legend(fontsize=25)
        plt.xlabel('epochs', fontsize=25)
        plt.ylabel('recall', fontsize=25)
        plt.title('Recalls', fontsize=25)
        filepath = os.path.join('./figures', self.name + '_rec' + '.png')
        plt.savefig(filepath)
        
        plt.figure(figsize=(10, 10))
        plt.plot(train_f1s, label='train')
        plt.plot(test_f1s, label='test')
        plt.legend(fontsize=25)
        plt.xlabel('epochs', fontsize=25)
        plt.ylabel('f1', fontsize=25)
        plt.title('F1s', fontsize=25)
        filepath = os.path.join('./figures', self.name + '_F1' + '.png')
        plt.savefig(filepath)
        plt.show()
        return 
    
    def plot_spike_patterns(self, test_spikes, net_ids):
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5*len(net_ids), 10))
        fig, ax = plt.subplots(1, 1, figsize=(3*len(net_ids), 5))
        plt.tight_layout()

        class1Spikes, class0Spikes = [], []
        labels = ['class i = {}'.format(str(id)) for id in net_ids]
        # labels = [str(id) for id in net_ids]

        for i, id in enumerate(net_ids):
            net = self.nets[id]
            class1_data, class0_data = [], []
            for x in test_spikes:
                x = torch.from_numpy(x).view(1, -1).to(self.device)
                s = x.T - self.times
                srf = self.srf(-1.0*s)
                class1_spikes, class0_spikes, _, _ = net.forward(srf)                
                class1_data.append(class1_spikes)
                class0_data.append(class0_spikes)
            assert len(class1_data) == len(test_spikes)
            assert len(class0_data) == len(test_spikes)
            class1Spikes.append(np.array(class1_data))
            class0Spikes.append(np.array(class0_data))

        # B1 = ax1.boxplot(class1Spikes, vert=True, patch_artist=True, labels=labels)
        B1 = ax.boxplot(class1Spikes, vert=True, patch_artist=True, labels=labels)
        for median in B1['medians']:
            median.set(color ='yellow', linewidth = 3)
        # ax1.set_title('Spike times for '+r'$S^+_i$ set', fontsize=15)
        ax.set_title(self.name+' dataset: '+'Spike times for '+r'$S^+_i$ set', fontsize=18)
        # B2 = ax2.boxplot(class0Spikes, vert=True, patch_artist=True, labels=labels)
        # ax2.set_title('Spike times for '+r'$S^-_i$ set', fontsize=15)
        # for b1, b2 in zip(B1['boxes'], B2['boxes']):
        #     b1.set_facecolor('green')
        #     b2.set_facecolor('blue')
        
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()
        return
        
        
## to implement - sklearn.metrics.hamming_loss, sklearn.metrics.zero_one_loss, sklearn.metrics.jaccard_score
## include number of classes to use for evaluation, also include the feature to combine training and testing datasets for evaluation
## our method is deterministic - the 2022 method is stochastic - so we are more safe in a way that we can give better and exact decisions, also our model is interpretable
## so we only need to use the metrics we computed above and not use additinoal metrics like average precision, coverage, ranking loss etc. which need posterior probabilities for evaluation. that reduces the number of metrics needed for evaluation

class Metrics:
    def __init__(self, model, train_data, train_labels, test_data, test_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = np.array(train_labels)
        self.test_data = test_data
        self.test_labels = np.array(test_labels)
        self.modes = ['train', 'test', 'combined']
        
    def evaluate(self):
        self.model.plot_results()
        self.predict_on_train()
        self.predict_on_test()
        print("Model was trained on {} classes".format(self.model.n_outputs))
        for mode in self.modes:
            print("Evaluating Model on {} mode".format(mode))
            self.evaluate_classwise_metrics(mode)
            self.evaluate_overall_metrics(mode)
            
    def continual_learning_evaluation(self, label_list, mode='test', continual_mode='individual'):
        print("Evaluating in {} mode on {} set".format(continual_mode, mode))
        self.predict_on_train()
        self.predict_on_test()
        cum_label_list = [0]
        for k in label_list:
            cum_label_list.append(cum_label_list[-1]+k)
        start_labels = cum_label_list[:-1]
        end_labels = cum_label_list[1:]
#         print(start_labels)
#         print(end_labels)
        
        if mode == 'train':
            y_true = self.train_labels
            y_pred = self.train_pred_labels
        elif mode == 'test':
            y_true = self.test_labels
            y_pred = self.test_pred_labels
        else:
            y_true = np.vstack([self.train_labels, self.test_labels])
            y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])
            
        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': []}
        
        for j in range(len(label_list)):
            if continual_mode == 'individual':
                start_label = start_labels[j]
            else:
                start_label = start_labels[0]
            end_label = end_labels[j]
            
            Y_true, Y_pred = y_true[:, start_label:end_label], y_pred[:, start_label:end_label]
#             ytj = np.zeros(Y_true.shape)
#             ytj[np.where(Y_true==1)] = 1
#             ypj = np.zeros(Y_pred.shape)
#             ypj[np.where(Y_pred==1)] = 1
            
#             print(type_of_target(ypj), type_of_target(ytj))
#             print(Y_true.dtype, Y_pred.dtype)
#             print(Y_true.shape, Y_pred.shape)
            
            results_dict['hamming loss'].append(hamming_loss(Y_true, Y_pred))
            results_dict['zero_one_loss'].append(zero_one_loss(Y_true, Y_pred))
            results_dict['one_error'].append(one_error(Y_true, Y_pred))
            results_dict['micro av. jaccard'].append(jaccard_score(Y_true, Y_pred, average='micro'))
            results_dict['macro av. jaccard'].append(jaccard_score(Y_true, Y_pred, average='macro'))
            results_dict['micro av. precision'].append(precision_score(Y_true, Y_pred, average='micro'))
            results_dict['macro av. precision'].append(precision_score(Y_true, Y_pred, average='macro'))
            results_dict['micro av. recall'].append(recall_score(Y_true, Y_pred, average='micro'))
            results_dict['macro av. recall'].append(recall_score(Y_true, Y_pred, average='macro'))
            results_dict['micro av. f1'].append(f1_score(Y_true, Y_pred, average='micro'))
            results_dict['macro av. f1'].append(f1_score(Y_true, Y_pred, average='macro'))
            results_df = pd.DataFrame.from_dict(results_dict)
            
        table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        print(results_df.T.to_markdown())
            
    def predict_on_train(self):
        pred_labels = []
#         true_labels = []
        for j in range(len(self.train_data)):
            x = torch.from_numpy(self.train_data[j]).view(1, -1).to(self.model.device)
            s = x.T - self.model.times
            srf = self.model.srf(-1.0*s)
            pred_label = []
#             true_labels.append(labels[j])
            for i, net in enumerate(self.model.nets):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label)
        
#         true_labels = np.array(self.train_labels)
        self.train_pred_labels = np.array(pred_labels)
        
    def predict_on_test(self):
        pred_labels = []
#         true_labels = []
        for j in range(len(self.test_data)):
            x = torch.from_numpy(self.test_data[j]).view(1, -1).to(self.model.device)
            s = x.T - self.model.times
            srf = self.model.srf(-1.0*s)
            pred_label = []
#             true_labels.append(labels[j])
            for i, net in enumerate(self.model.nets):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label)
        
#         true_labels = np.array(self.test_labels)
        self.test_pred_labels = np.array(pred_labels)
    
    def evaluate_classwise_metrics(self, mode):
        print("Classwise Results")
        if mode == 'train':
            y_true = self.train_labels
            y_pred = self.train_pred_labels
        elif mode == 'test':
            y_true = self.test_labels
            y_pred = self.test_pred_labels
        else:
            y_true = np.vstack([self.train_labels, self.test_labels])
            y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])
        results_dict = {'class': [], 'accuracy': [], 'hamming loss': [], 'zero_one_loss': [], 'micro av. jaccard': [], 'macro av. jaccard': [], 'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': []}
        for i in range(self.model.n_outputs):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            results_dict['class'].append(i+1) 
            results_dict['accuracy'].append(accuracy_score(y_true_i, y_pred_i))
            results_dict['hamming loss'].append(hamming_loss(y_true_i, y_pred_i))
            results_dict['zero_one_loss'].append(zero_one_loss(y_true_i, y_pred_i))
            results_dict['micro av. jaccard'].append(jaccard_score(y_true_i, y_pred_i, average='micro'))
            results_dict['macro av. jaccard'].append(jaccard_score(y_true_i, y_pred_i, average='macro'))
            results_dict['micro av. precision'].append(precision_score(y_true_i, y_pred_i, average='micro'))
            results_dict['macro av. precision'].append(precision_score(y_true_i, y_pred_i, average='macro'))
            results_dict['micro av. recall'].append(recall_score(y_true_i, y_pred_i, average='micro'))
            results_dict['macro av. recall'].append(recall_score(y_true_i, y_pred_i, average='macro'))
            results_dict['micro av. f1'].append(f1_score(y_true_i, y_pred_i, average='micro'))
            results_dict['macro av. f1'].append(f1_score(y_true_i, y_pred_i, average='macro'))
        
        results_df = pd.DataFrame.from_dict(results_dict)
        table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        print(results_df.T.to_markdown())
        
    def evaluate_overall_metrics(self, mode):
        print("Overall Results")
        if mode == 'train':
            y_true = self.train_labels[:, :self.model.n_outputs]
            y_pred = self.train_pred_labels[:, :self.model.n_outputs]
        elif mode == 'test':
            y_true = self.test_labels[:, :self.model.n_outputs]
            y_pred = self.test_pred_labels[:, :self.model.n_outputs]
        else:
            y_true = np.vstack([self.train_labels[:, :self.model.n_outputs], self.test_labels[:, :self.model.n_outputs]])
            y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])
        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': []}
        results_dict['hamming loss'].append(hamming_loss(y_true, y_pred))
        results_dict['zero_one_loss'].append(zero_one_loss(y_true, y_pred))
        results_dict['one_error'].append(one_error(y_true, y_pred))
        results_dict['micro av. jaccard'].append(jaccard_score(y_true, y_pred, average='micro'))
        results_dict['macro av. jaccard'].append(jaccard_score(y_true, y_pred, average='macro'))
        results_dict['micro av. precision'].append(precision_score(y_true, y_pred, average='micro'))
        results_dict['macro av. precision'].append(precision_score(y_true, y_pred, average='macro'))
        results_dict['micro av. recall'].append(recall_score(y_true, y_pred, average='micro'))
        results_dict['macro av. recall'].append(recall_score(y_true, y_pred, average='macro'))
        results_dict['micro av. f1'].append(f1_score(y_true, y_pred, average='micro'))
        results_dict['macro av. f1'].append(f1_score(y_true, y_pred, average='macro'))
        results_df = pd.DataFrame.from_dict(results_dict)
        table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        print(results_df.T.to_markdown())
        
        
    