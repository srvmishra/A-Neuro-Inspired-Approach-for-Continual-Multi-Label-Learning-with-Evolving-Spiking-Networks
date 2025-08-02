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
from utils import *
from continual_methods import *
warnings.filterwarnings("ignore")

### only debugging and running are left

def process_data(trainpath, testpath, num_features):
    if trainpath.endswith('.arff'):
        train_ = arff.load(open(trainpath, 'rt'))
        train_ = np.array(train_['data']).astype(np.float32)
        train_data = train_[:, :num_features]
        train_labels = train_[:, num_features:]
    if testpath.endswith('.arff'):
        test_ = arff.load(open(testpath, 'rt'))
        test_ = np.array(test_['data']).astype(np.float32)
        test_data = test_[:, :num_features]
        test_labels = test_[:, num_features:]

    if trainpath.endswith('.mat'):
        train_ = loadmat(trainpath)
        train_data, train_labels = train_['transformed_train_data'], np.array(train_['labels'])
    if testpath.endswith('.mat'):
        test_ = loadmat(testpath)
        test_data, test_labels = test_['transformed_test_data'], np.array(test_['labels'])

    print(train_data.shape)
    print(test_data.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    return train_data, train_labels, test_data, test_labels

def normal_train(model, trainData, augTrainLabels, optimizer, criterion, epochs, device):
    model.train().to(self.device)
    for epoch in tqdm(range(epochs)):
        for i in range(len(self.data)):
            x = self.data[i]
            y = self.labels[i]
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).float().to(self.device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    return model

class Experiment:
    def __init__(self, hparams):
        # model parameters
        self.input_dim = hparams['input_dim']
        self.hidden_dim = hparams['hidden_dim']
#         self.out_dim = hparams['out_dim']
        self.device = hparams['device']
        self.ckpt_dir = hparams['ckpt_dir']

        # dataset parameters
        self.name = hparams['name']
        self.train_path = hparams['trainpath']
        self.test_path = hparams['testpath']
        self.num_features = hparams['num_features']
        self.seed = hparams['seed']
        self.tasks = hparams['tasks']

        # optimizer parameters
        self.lr = hparams['lr']
        self.epochs = hparams['epochs']
        self.loss_type = hparams['loss_type']

        # continual learning mode
        self.mode = hparams['continual_mode']['name']
        self.continual_params = hparams['continual_mode']['cparams']
        self.measure = 'micro av. f1'

        self.generator = np.random.RandomState(self.seed)
#         self.model = MLP(self.input_dim, self.hidden_dim, self.out_dim)
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
        # add a few lines about loss functions
        if self.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        if self.loss_type == 'asy':
            self.criterion = AsymmetricLossOptimized()
        if self.loss_type == 'corr':
            self.criterion = CorrelationLoss(self.device)
        if self.loss_type == 'both':
            self.criterion = CorrelationAsymmetricLoss(self.device)

        pass

    def augment_labels(self, trainData, trainLabels, model):
        pred_labels = []
        for j in range(len(trainData)):
            x = torch.from_numpy(trainData[j]).to(self.device).float()
            yhat = model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        pred_labels = np.array(pred_labels)
        augTrainLabels = np.hstack([pred_labels, trainLabels])
        return augTrainLabels

    def run(self):
        print('Processing Data ...')
        train_data, train_labels, test_data, test_labels = process_data(self.train_path, self.test_path, self.num_features)
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
            end_samples = [len(train_data)]*len(labels_list)
        
        cum_label_list = [0]
        for k in labels_list:
            cum_label_list.append(cum_label_list[-1]+k)
        start_labels = cum_label_list[:-1]
        end_labels = cum_label_list[1:]
        
        tasks = len(labels_list)

        print('Begin training in {} mode ...'.format(self.mode))
        for task in range(tasks):
            print("Training and Evaluating on task {}".format(task+1))
            start_label = start_labels[task]
            end_label = end_labels[task]
            start_sample = start_samples[task]
            end_sample = end_samples[task]
            
            trainData = train_data[start_sample:end_sample, :]
            if self.seed != 2:
                indices = self.generator.permutation(train_labels.shape[1])
                train_labels = train_labels[:, indices]
            trainLabels = train_labels[start_sample:end_sample, start_label:end_label]
            print("Train data shape is ", trainData.shape)
            print("Train Labels shape is ", trainLabels.shape)
            print("Classes trained are: {} to {}".format(start_label, end_label))
            
            if task > 0:
                print("Augmenting class labels...")
                augTrainLabels = self.augment_labels(trainData, trainLabels, model)
                print("Train Labels shape is ", augTrainLabels.shape)
                out_dim = augTrainLabels.shape[1]
                print("Output dimensions: {}".format(out_dim))
                
                if self.mode == 'EWC':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    optimizer = optim.SGD(model.parameters(), lr=self.lr)
                    ewc = EWC(model, trainData, augTrainLabels, self.device, self.criterion, optimizer, self.continual_params['importance'])
                    for epoch in tqdm(range(epochs)):
                        model = ewc.train(model)
                        
                if self.mode == 'LwF':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    optimizer = optim.SGD(model.parameters(), lr=self.lr)
                    lwf = LwF(prev_model, trainData, augTrainLabels, self.device, optimizer, self.continual_params['importance'])
                    for epoch in tqdm(range(epochs)):
                        model = lwf.train(model)
                    prev_model = model.copy()
                
                if self.mode == 'SI':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    optimizer = optim.SGD(model.parameters(), lr=self.lr)
                    si = SI(model, trainData, augTrainLabels, self.device, self.criterion, optimizer, self.continual_params['importance'])
                    for epoch in tqdm(range(epochs)):
                        model = si.train(model)
                        si.change_model(model)
                        optimizer = optim.SGD(model.parameters(), lr=self.lr)
                        si.change_optimizer(optimizer)
                        
            else:
                augTrainLabels = trainLabels
                out_dim = trainLabels.shape[1]
                print("Output dimensions: {}".format(out_dim))
                model = MLP(self.input_dim, self.hidden_dim, out_dim)
                optimizer = optim.SGD(model.parameters(), lr=self.lr)
                model = normal_train(model, trainData, augTrainLabels, optimizer, self.criterion, self.epochs, self.device)
                prev_model = model.copy()            
            
            metric = Metrics(model, self.device, trainData, trainLabels, test_data, test_labels)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='individual')
            self.results['individual'].append(results)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='combined')
            self.results['combined'].append(results)
            
        metric = Metrics(model, self.device, trainData, trainLabels, test_data, test_labels)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='individual')
        self.results['final individual'].append(results)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='combined')
        self.results['final combined'].append(results)
        self.save(model)
        self.forgetting_stats()
        return
    
    def save(self, model):
        filename = self.name + '_' + str(self.seed) + '_' + self.mode + '.pkl'
        filepath = os.path.join(self.ckpt_dir, filename)
        # change here -- 
        # get model to cpu
        # save the state dict, and add the results field to it
        # then pickle and save it.
        with open(filepath, 'wb') as f:
            pickle.dump([model, self.results], f)
        f.close()
        print("File saved to {}".format(filepath))
        return        

    def forgetting_stats(self):
        n_tasks = len(self.tasks['labels'])
        print("Number of Tasks: {}".format(n_tasks))
        
        forgets = []
        for i in range(n_tasks):
            f1_task = self.results['individual'][i][self.measure][0]
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