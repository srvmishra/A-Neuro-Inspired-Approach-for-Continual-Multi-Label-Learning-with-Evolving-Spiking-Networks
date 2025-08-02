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
from hparams import *
from utils import *
from copy import deepcopy
warnings.filterwarnings("ignore")

class ANNUpperBoundExperiment:
    def __init__(self, hparams):
        self.hparams = hparams

        self.model = ANNModel(self.hparams.num_inputs, self.hparams.num_outputs, self.hparams.num_hidden_layers, 
                              self.hparams.num_neurons_per_layer)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def run(self):
        print("###################### Running on {} dataset ####################".format(self.name))
        torch.cuda.manual_seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        self.model, self.results = trainANN(self.model, self.hparams.name, self.hparams.train_path, self.hparams.test_path, 
                                            self.hparams.num_features, self.optimizer, self.hparams.criterion, self.hparams.epochs, 
                                            self.hparams.batch_size, self.hparams.device)
        self.save()
        return 
    
    def save(self):
        filename = self.hparams.name + '_' + str(self.hparams.seed) + '_upperbound_' + self.hparams.loss_type + '.pkl'
        filepath = os.path.join(self.hparams.ckpt_dir, filename)
        
        cpu = torch.device('cpu')
        self.model.to(cpu)
        checkpoint = {'model_state_dict': self.model.state_dict(), 'results': self.results}
        torch.save(checkpoint, filepath)
        print("File saved to {}".format(filepath))
        self.results = None
        return


class ContinualExperiment:
    def __init__(self, hparams):
        self.hparams = hparams        

    def augment_labels(self, trainData, trainLabels, model):
        pred_labels = []
        for j in range(len(trainData)):
            x = torch.from_numpy(trainData[j]).to(self.hparams.device).float()
            yhat = model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        pred_labels = np.array(pred_labels)
        augTrainLabels = np.hstack([pred_labels, trainLabels])
        return augTrainLabels

    def run(self):
        print('Processing Data ...')
        train_data, train_labels, test_data, test_labels = process_data(self.hparams.name, self.hparams.train_path, 
                                                                        self.hparams.test_path, self.hparams.num_features)
        samples_list = self.hparams.tasks['samples']
        labels_list = self.hparams.tasks['labels']
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

        print('Begin training in {} mode ...'.format(self.hparams.mode))
        for task in range(tasks):
            print("Training and Evaluating on task {}".format(task+1))
            start_label = start_labels[task]
            end_label = end_labels[task]
            start_sample = start_samples[task]
            end_sample = end_samples[task]
            
            trainData = train_data[start_sample:end_sample, :]
            if self.hparams.seed != 2:
                indices = self.hparams.generator.permutation(train_labels.shape[1])
                train_labels = train_labels[:, indices]
            trainLabels = train_labels[start_sample:end_sample, start_label:end_label]
            print("Train data shape is ", trainData.shape)
            print("Train Labels shape is ", trainLabels.shape)
            print("Classes trained are: {} to {}".format(start_label, end_label))
            print("Samples with all negative labels {}".format(num_zeros(trainLabels)))
            
            if task > 0:
                print("Augmenting class labels...")
                augTrainLabels = self.augment_labels(trainData, trainLabels, model)
                print("Train Labels shape is ", augTrainLabels.shape)
                out_dim = augTrainLabels.shape[1]
                print("Output dimensions: {}".format(out_dim))
                
                if self.hparams.mode == 'EWC':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    model.to(self.hparams.device)
                    optimizer = optim.Adam(model.parameters(), lr=self.hparams.lr)
                    ewc = EWC(model, trainData, augTrainLabels, self.hparams.device, self.hparams.criterion, optimizer, 
                              self.hparams.continual_params['importance'], self.hparams.batch_size)
                    for epoch in tqdm(range(self.hparams.epochs)):
                        model = ewc.train(model)
                        
                if self.hparams.mode == 'LwF':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    model.to(self.hparams.device)
                    optimizer = optim.Adam(model.parameters(), lr=self.hparams.lr)
                    lwf = LwF(prev_model, trainData, augTrainLabels, self.hparams.device, optimizer, 
                              self.hparams.continual_params['importance'], self.hparams.batch_size)
                    for epoch in tqdm(range(self.hparams.epochs)):
                        model = lwf.train(model)
                    prev_model = deepcopy(model)
                
                if self.hparams.mode == 'SI':
                    print("growing model size ...")
                    model.grow_size(out_dim)
                    model.to(self.hparams.device)
                    optimizer = optim.SGD(model.parameters(), lr=self.hparams.lr)
                    si = SI(model, trainData, augTrainLabels, self.hparams.device, self.hparams.criterion, optimizer, 
                            self.hparams.continual_params['importance'], self.hparams.batch_size)
                    for epoch in tqdm(range(self.hparams.epochs)):
                        model = si.train(model)
                        si.change_model(model)
                        optimizer = optim.Adam(model.parameters(), lr=self.hparams.lr)
                        si.change_optimizer(optimizer)
                        
            else:
                augTrainLabels = trainLabels
                out_dim = trainLabels.shape[1]
                print("Output dimensions: {}".format(out_dim))
                # model = MLP(self.input_dim, self.hidden_dim, out_dim)
                model = ANNModel(self.hparams.num_inputs, out_dim, self.hparams.num_hidden_layers, 
                                 self.hparams.num_neurons_per_layer)
                optimizer = optim.Adam(model.parameters(), lr=self.hparams.lr)
                model = normal_train_batch(model, trainData, augTrainLabels, optimizer, self.hparams.criterion, 
                                           self.hparams.epochs, self.hparams.device, self.hparams.batch_size)
                prev_model = deepcopy(model)            
            
            metric = NEWMetrics(model, self.hparams.device, trainData, trainLabels, test_data, test_labels)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='individual')
            self.results['individual'].append(results)
            results = metric.continual_learning_evaluation(labels_list, task_id=task, mode='test', continual_mode='combined')
            self.results['combined'].append(results)
            
        metric = NEWMetrics(model, self.hparams.device, trainData, trainLabels, test_data, test_labels)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='individual')
        self.results['final individual'].append(results)
        results = metric.continual_learning_evaluation(labels_list, mode='test', continual_mode='combined')
        self.results['final combined'].append(results)
        self.save(model)
        self.forgetting_stats()
        return
    
    def save(self, model):
        filename = self.hparams.name + '_' + str(self.hparams.seed) + '_' + self.hparams.mode + '_' + self.hparams.loss_type + '.pkl'
        filepath = os.path.join(self.hparams.ckpt_dir, filename)
        
        cpu = torch.device('cpu')
        model.to(cpu)
        checkpoint = {'model_state_dict': model.state_dict(), 'results': self.results}
        torch.save(checkpoint, filepath)
        print("File saved to {}".format(filepath))

        ## loading the model for further stuff in a function elsewhere
        # model = TheModelClass(*args, **kwargs)
        # optimizer = TheOptimizerClass(*args, **kwargs)

        # checkpoint = torch.load(PATH)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # results = checkpoint['results'] 
        # ...
        return        

    def forgetting_stats(self):
        n_tasks = len(self.hparams.tasks['labels'])
        print("Number of Tasks: {}".format(n_tasks))
        
        forgets = []
        for i in range(n_tasks):
            f1_task = self.results['individual'][i][self.hparams.measure][0]
            f1_combined = self.results['final individual'][0][self.hparams.measure][0]
            forgets.append(f1_task-f1_combined)
            print("Forgetting in task {}: first time {} = {:.4f}, after all tasks {} = {:.4f}".format(i+1, self.hparams.measure, f1_task, 
                                                                                                      self.hparams.measure, f1_combined))
            
        plt.figure(figsize=(5, 5))
        plt.bar(list(range(n_tasks)), forgets)
        plt.xlabel('tasks')
        plt.ylabel('forgetting')
        plt.title('Forgetting in {} dataset'.format(self.hparams.name))
        plt.show()
        return