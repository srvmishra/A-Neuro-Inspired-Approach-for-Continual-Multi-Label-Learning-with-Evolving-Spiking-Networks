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
    # print(data.shape)
    means = torch.Tensor([(2*h-3)/(2*(num_gaussians-2)) for h in range(1, num_gaussians+1)]).to(device).float()
    std_dev = 1.0/(0.7*(num_gaussians-2))
    normalize = lambda x, a, b: (x - b)/(a - b)
#     a = data[:, :num_features].max(axis=0)
#     b = data[:, :num_features].min(axis=0)
    data[:, :num_features] = normalize(data[:, :num_features], a, b)
    T = 300
    spiked_matrices = []
#     spiked_labels = []
    for i in tqdm(range(len(data))):
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
    return spiked_matrices, np.array(spiked_labels)

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
                
        class1_pos = [i for i in range(len(self.nature)) if self.nature[i] == 1]
        class0_pos = [i for i in range(len(self.nature)) if self.nature[i] == 0]
        
    def add_extra_neuron(self, srf, nature):
        v = srf[:, self.third_time].view(-1, 1).to(self.device)
        norm_srf = srf/srf.sum(0)
        u = norm_srf[:, self.third_time].view(1, -1).to(self.device)
        
        if len(self.weights) > 0:
            self.weights = torch.cat([self.weights, u], dim=0)
            self.theta = torch.cat([self.theta, torch.matmul(u, v)], dim=0)
            self.nature.append(nature)
        else:
            self.weights = u
            self.theta = torch.matmul(u, v).view(-1, 1)   
            self.nature.append(nature)
                
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
        self.name = hparams['name']
        self.tasks = hparams['tasks']
        self.load_from_file = hparams['load_from_file']
        if 'add_extra_neuron' not in hparams.keys():
            self.add_extra_neuron = False
        else:
            self.add_extra_neuron = hparams['add_extra_neuron']
        self.times = torch.linspace(0, hparams['sim_time'], hparams['sim_time']+1).unsqueeze(0).to(self.device)
        self.generator = np.random.RandomState(self.seed)
        self.measure = 'micro av. f1'
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
    
    def neuron_addition_operation(self, train_spikes, train_labels, nets_to_train=None):
        if nets_to_train is not None:
            nets = nets_to_train
        else:
            nets = self.nets
        for j in range(len(train_spikes)):
            x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            for i, net in enumerate(nets):
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
        for i, net in enumerate(nets):
            c1 = [i for i in range(len(net.nature)) if net.nature[i] == 1]
            c0 = [i for i in range(len(net.nature)) if net.nature[i] == 0]
            
            if self.add_extra_neuron:
                if len(c1) == 0:
                    j = self.generator.choice(np.array(list(range(len(train_spikes)))))
                    x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
                    s = x.T - self.times
                    srf = self.srf(-1.0*s)
                    net.add_extra_neuron(srf, 1)
                if len(c0) == 0:
                    j = self.generator.choice(np.array(list(range(len(train_spikes)))))
                    x = torch.from_numpy(train_spikes[j]).view(1, -1).to(self.device)
                    s = x.T - self.times
                    srf = self.srf(-1.0*s)
                    net.add_extra_neuron(srf, 0)
                    
            c1 = [i for i in range(len(net.nature)) if net.nature[i] == 1]
            c0 = [i for i in range(len(net.nature)) if net.nature[i] == 0]
            
            print("Class {}, class 1 neurons = {}, class 0 neurons = {}".format(i, len(c1), len(c0)))
            
    def fit_on_task(self, trainData, trainLabels, nets_to_train):
        indices = np.array([k for k in range(len(trainData))])
        self.neuron_addition_operation(trainData, trainLabels, nets_to_train=nets_to_train)
        for k in tqdm(range(self.epochs)): 
            self.generator.shuffle(indices)
            for j in indices:
                x = torch.from_numpy(trainData[j]).view(1, -1).to(self.device)
                s = x.T - self.times
                srf = self.srf(-1.0*s)
                for i, net in enumerate(nets_to_train):
                    y = trainLabels[j, i] 
                    if y == 1:
                        nature = 'cc'
                        cc = 1
                        oc = 0
                    else:
                        nature = 'oc'
                        cc = 0
                        oc = 1
                    net.update(srf, cc)
        return
    
    def augment_labels(self, trainData, trainLabels, nets_to_train):
        pred_labels = []
        for j in range(len(trainData)):
            x = torch.from_numpy(trainData[j]).view(1, -1).to(nets_to_train[0].device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            pred_label = []
#             true_labels.append(labels[j])
            for i, net in enumerate(nets_to_train):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label)
        pred_labels = np.array(pred_labels).reshape((len(pred_labels), -1))
        augLabels = np.hstack([pred_labels, trainLabels])
        return augLabels
    
    def fit_and_evaluate(self, train_spikes, train_labels, test_spikes, test_labels):
        samples_list = self.tasks['samples']
        labels_list = self.tasks['labels']
        self.results = {'individual': [], 'combined': [], 'final individual': [], 'final combined': []}
        
        if samples_list is not None:
            cum_sample_list = [0]
            for k in samples_list:
                cum_sample_list.append(cum_sample_list[-1]+k)
            start_samples = cum_sample_list[:-1]
            end_samples = cum_sample_list[1:]
        else:
            start_samples = [0]*len(labels_list)
            end_samples = [len(train_spikes)]*len(labels_list)
        
        cum_label_list = [0]
        for k in labels_list:
            cum_label_list.append(cum_label_list[-1]+k)
        start_labels = cum_label_list[:-1]
        end_labels = cum_label_list[1:]
        
        tasks = len(labels_list)
        
        if self.seed != 2:
            indices = self.generator.permutation(train_labels.shape[1])
            train_labels = train_labels[:, indices]
            test_labels = test_labels[:, indices]
        
        for task in range(tasks):
            print("Training and Evaluating on task {}".format(task+1))
            start_label = start_labels[task]
            end_label = end_labels[task]
            start_sample = start_samples[task]
            end_sample = end_samples[task]
            
            trainData = train_spikes[start_sample:end_sample]
            trainLabels = train_labels[start_sample:end_sample, start_label:end_label]
            print("Train data shape is ", np.array(trainData).shape)
            print("Train Labels shape is ", np.array(trainLabels).shape)
            print("Classes trained are: {} to {}".format(start_label, end_label))
            if task > 0:
                print("Augmenting class labels...")
                augTrainLabels = self.augment_labels(trainData, trainLabels, nets_to_train)
                print("Train Labels shape is ", augTrainLabels.shape)
            else:
                augTrainLabels = trainLabels
            nets_to_train = self.nets[0:end_label]
            self.fit_on_task(trainData, augTrainLabels, nets_to_train)
            
            metric = Metrics(nets_to_train, self.tau, self.times, trainData, trainLabels, test_spikes, test_labels)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='individual')
            self.results['individual'].append(results)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='combined')
            self.results['combined'].append(results)
            
        metric = Metrics(self.nets, self.tau, self.times, train_spikes, train_labels, test_spikes, test_labels)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='individual')
        self.results['final individual'].append(results)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='combined')
        self.results['final combined'].append(results)
        self.save()
        self.forgetting_stats()
        return 
    
    def save(self):
        if self.tasks['samples'] is not None:
            mode = 'cifdm'
        else:
            mode = 'dsll'
        filename = self.name + '_' + str(self.seed) + '_' + mode + '.pkl'
        filepath = os.path.join(self.ckpt_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump([self.nets, self.results], f)
        f.close()
#         print("File saved to {}".format(filepath))
        return        

    def forgetting_stats(self):
        n_tasks = len(self.tasks['labels'])
        print("Number of Tasks: {}".format(n_tasks))
        
        forgets = []
        for i in range(n_tasks):
            f1_task = self.results['individual'][i][self.measure][0]
            # small mistake - put i at the last index in the next line
            f1_combined = self.results['final individual'][0][self.measure][0]
            forgets.append(f1_task-f1_combined)
            print("Forgetting in task {}: first time {} = {:.4f}, after all tasks {} = {:.4f}".format(i+1, self.measure, f1_task, self.measure, f1_combined))
            
        plt.figure(figsize=(5, 5))
        plt.bar(list(range(n_tasks)), forgets)
        plt.xlabel('tasks')
        plt.ylabel('forgetting')
        plt.title('Forgetting in {} dataset'.format(self.name))
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
        spk_diff = [a-b for a,b in zip(class1Spikes, class0Spikes)]
        B1 = ax.boxplot(spk_diff, vert=True, patch_artist=True, labels=labels)
        for median in B1['medians']:
            median.set(color ='yellow', linewidth = 3)
        # ax1.set_title('Spike times for '+r'$S^+_i$ set', fontsize=15)
        ax.set_title(self.name+' dataset: '+'Spike time differences', fontsize=18)
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
    def __init__(self, nets, tau, times, train_data, train_labels, test_data, test_labels):
        self.nets = nets
        self.tau = tau
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.times = times
        self.modes = ['train', 'test', 'combined']
        
    def srf(self, s):
        f = s*torch.exp(1.0 - s/self.tau)/self.tau
        f[torch.where(s<=0)] = 0.0
        return f.double()
            
    def continual_learning_evaluation(self, label_list, task_id=None, mode='test', continual_mode='individual'):
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
        
        if task_id is not None:
            if continual_mode == 'individual':
                start_label = start_labels[task_id]
            else:
                start_label = start_labels[0]
            end_label = end_labels[task_id]
            Y_true, Y_pred = y_true[:, start_label:end_label], y_pred[:, start_label:end_label]
            print(type_of_target(Y_true), type_of_target(Y_pred))
            print(Y_true.dtype, Y_pred.dtype)
            print(Y_true.shape, Y_pred.shape)
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
        else:
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
        return results_dict
            
    def predict_on_train(self):
        pred_labels = []
#         true_labels = []
        for j in range(len(self.train_data)):
            x = torch.from_numpy(self.train_data[j]).view(1, -1).to(self.nets[0].device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            pred_label = []
#             true_labels.append(labels[j])
            for i, net in enumerate(self.nets):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label)
        
#         true_labels = np.array(self.train_labels)
        self.train_pred_labels = np.array(pred_labels)
        
    def predict_on_test(self):
        pred_labels = []
#         true_labels = []
        for j in range(len(self.test_data)):
            x = torch.from_numpy(self.test_data[j]).view(1, -1).to(self.nets[0].device)
            s = x.T - self.times
            srf = self.srf(-1.0*s)
            pred_label = []
#             true_labels.append(labels[j])
            for i, net in enumerate(self.nets):
                pred = net.predict(srf)
                pred_label.append(pred)
            pred_labels.append(pred_label)
        
#         true_labels = np.array(self.test_labels)
        self.test_pred_labels = np.array(pred_labels)
    
        
    