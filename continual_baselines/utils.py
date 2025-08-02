import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, zero_one_loss, hamming_loss, jaccard_score
from sklearn.utils.multiclass import type_of_target
import torch
import arff
from scipy.io import loadmat, savemat
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import markdown

# Contents of this file
# model class - done
# metric class - done
# loss functions - done
# LwF - done
# EWC - done
# SI - done

# main file for train and test
# create dataloaders also - done
# grow model size - done
# add pseudo labels eveytime - done
# create arguments and check - done
# add loss functions - done

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=8, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

class CorrelationLoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(CorrelationLoss, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, pred, label):
        if len(label.shape) == 1:
            pred = label.unsqueeze(0)

        n_one = torch.sum(label, 1)
        n_zero = torch.ones(label.shape[0]).to(self.device) * label.shape[1]
        n_zero -= n_one

        result_matrix = torch.zeros(pred.shape).to(self.device)

        temp_result = torch.exp(pred - 1)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_n = n_zero + (n_zero == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_one == 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, (1-label))
        result_matrix += temp_result

        temp_result = torch.exp(-pred)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_n = n_one + (n_one == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_zero == 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, label)
        result_matrix += temp_result

        temp_result = torch.transpose(torch.matmul(torch.ones([label.shape[1], 1]).to(self.device), torch.unsqueeze(pred, 1)), 1, 2)
        temp_minus = torch.matmul(torch.ones([label.shape[1], 1]).to(self.device), torch.unsqueeze(pred, 1))
        temp_result = torch.exp(temp_minus - temp_result) * torch.unsqueeze(1-label, 1)
        temp_result = temp_result * torch.transpose(torch.unsqueeze(label, 1), 1, 2)
        temp_result = torch.sum(temp_result, 2)
        temp_result = torch.transpose(temp_result, 1, 0)
        n_else = n_one * n_zero
        temp_n = n_else + (n_else == 0).float()
        temp_result = torch.div(temp_result, temp_n)
        temp_mask = (n_else != 0).float()
        temp_result = torch.mul(temp_result, temp_mask)
        temp_result = torch.transpose(temp_result, 1, 0)
        temp_result = torch.mul(temp_result, label)
        result_matrix += temp_result

        result_matrix *= ((self.weight - 1) * label) + 1

        return torch.sum(result_matrix)
    
class CorrelationAsymmetricLoss(torch.nn.Module):
    def __init__(self, device, gamma_pos=6, weight=1):
        super(CorrelationAsymmetricLoss, self).__init__()
        self.asy = AsymmetricLossOptimized(gamma_pos)
        self.correlation = CorrelationLoss(device, weight)
    def forward(self, pred, label):
        return self.asy(pred, label) + self.correlation(pred, label)

def one_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    err = 0
    for i in range(y_true.shape[0]):
        err = err + (y_true[i, :] != y_pred[i, :]).all()
    val = err/y_true.shape[0]
    return val

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.layer3 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.layer3(x)
        return x
    
    def grow_size(self, new_size):
        self.layer3 = nn.Linear(self.hidden_dim, new_size)
        self.out_dim = new_size
        return


## use this model in the new experiments class
class ANNModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers=None, num_neurons_per_layer=None):
        super(ANNModel, self).__init__()
        self.num_inputs = num_inputs
        self.out_dim = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        # self.copy_weights = copy_weights

        if self.num_hidden_layers is not None and self.num_neurons_per_layer is not None:
            layers = []
            layers.append(nn.Linear(self.num_inputs, self.num_neurons_per_layer))
            layers.append(nn.ReLU())
            for i in range(self.num_hidden_layers):
                layers.append(nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer))
                layers.append(nn.ReLU())
            
            self.feature_extractor = nn.Sequential(*layers)
            self.clf = nn.Linear(self.num_neurons_per_layer, self.out_dim)
        else:
            self.feature_extractor = None
            self.clf = nn.Linear(self.num_inputs, self.out_dim)
            
    def forward(self, x):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        x = self.clf(x)
        return x

    def grow_size(self, new_size):
        if self.num_neurons_per_layer is not None:
            new_layer = nn.Linear(self.num_neurons_per_layer, new_size)
        else:
            new_layer = nn.Linear(self.num_inputs, new_size)

        self.clf = new_layer
        self.out_dim = new_size






    ## effect of retaining previous parameters in baseline experiments for CVPR
    
class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
def create_dataloader(data, labels, batch_size):
    dataset = MultiLabelDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size)

def process_data(name, trainpath, testpath, num_features):
    if name == 'FoodTruck':
    # if name == 'foodtruck'
        train_data = arff.load(open(trainpath, 'rt'))
        col_names = [x[0] for x in train_data['attributes']]
        train_data_arr = np.array(train_data['data'])
        train_data_arr = np.vstack([np.array(col_names), train_data_arr])
        np.savetxt('../../datasets/foodtruck/train.csv', train_data_arr, fmt='%s', delimiter=',')
        train_df = pd.read_csv('../../datasets/foodtruck/train.csv')
        train_df['time'].replace(['lunch', 'afternoon', 'happy_hour', 'dinner', 'dawn'], [0, 1, 2, 3, 4], inplace=True)
        train_df['motivation'].replace(['ads', 'by_chance', 'friend', 'social_network', 'web'], [0, 1, 2, 3, 4], inplace=True)
        train_df['marital.status'].replace(['divorced', 'married', 'single'], [0, 1, 2], inplace=True)
        train_df['gender'].replace(['F', 'M'], [0, 1], inplace=True)
        for c in train_df.columns:
            train_df[c] = train_df[c].astype('float')
        train_ = train_df.to_numpy()
        train_data = train_[:, :num_features]
        train_labels = train_[:, num_features:]

        test_data = arff.load(open(testpath, 'rt'))
        col_names = [x[0] for x in test_data['attributes']]
        test_data_arr = np.array(test_data['data'])
        test_data_arr = np.vstack([np.array(col_names), test_data_arr])
        np.savetxt('../../datasets/foodtruck/test.csv', train_data_arr, fmt='%s', delimiter=',')
        test_df = pd.read_csv('../../datasets/foodtruck/test.csv')
        test_df['time'].replace(['lunch', 'afternoon', 'happy_hour', 'dinner', 'dawn'], [0, 1, 2, 3, 4], inplace=True)
        test_df['motivation'].replace(['ads', 'by_chance', 'friend', 'social_network', 'web'], [0, 1, 2, 3, 4], inplace=True)
        test_df['marital.status'].replace(['divorced', 'married', 'single'], [0, 1, 2], inplace=True)
        test_df['gender'].replace(['F', 'M'], [0, 1], inplace=True)
        for c in test_df.columns:
            test_df[c] = test_df[c].astype('float')
        test_ = test_df.to_numpy()
        test_data = test_[:, :num_features]
        test_labels = test_[:, num_features:]

    if name == 'Eukaryote':
        train_ = loadmat(trainpath)
        test_ = loadmat(testpath)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_labels = np.delete(train_labels, [0, 12, 15], 1)
        test_labels = np.delete(test_labels, [0, 12, 15], 1)

    if name == 'Human':
        train_ = loadmat(trainpath)
        test_ = loadmat(testpath)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_labels = np.delete(train_labels, [3, 8, 13], 1)
        test_labels = np.delete(test_labels, [3, 8, 13], 1)

    if trainpath.endswith('.arff') and name != 'FoodTruck':
        train_ = arff.load(open(trainpath, 'rt'))
        train_ = np.array(train_['data']).astype(np.float32)
        train_data = train_[:, :num_features]
        train_labels = train_[:, num_features:]
    if testpath.endswith('.arff') and name != 'FoodTruck':
        test_ = arff.load(open(testpath, 'rt'))
        test_ = np.array(test_['data']).astype(np.float32)
        test_data = test_[:, :num_features]
        test_labels = test_[:, num_features:]

    if trainpath.endswith('.mat') and name not in ['Eukaryote', 'Human']:
        train_ = loadmat(trainpath)
        train_data, train_labels = train_['data'], np.array(train_['labels'])
    if testpath.endswith('.mat') and name not in ['Eukaryote', 'Human']:
        test_ = loadmat(testpath)
        test_data, test_labels = test_['data'], np.array(test_['labels'])

    if name == 'Yeast':
        train_labels = train_labels[:, :-1]
        test_labels = test_labels[:, :-1]

    print(train_data.shape)
    print(test_data.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    return train_data, train_labels, test_data, test_labels

def num_zeros(data):
    nlabels = data.shape[1]
    num_zeros = 0
    for i in range(nlabels):
        if (data[:, i] == 0).all():
            num_zeros += 1
    return num_zeros

def normal_train(model, trainData, augTrainLabels, optimizer, criterion, epochs, device):
    model.train().to(device)
    for epoch in tqdm(range(epochs)):
        for i in range(len(trainData)):
            x = trainData[i]
            y = augTrainLabels[i]
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    return model

def normal_train_batch(model, trainData, augTrainLabels, optimizer, criterion, epochs, device, batch_size):
    model.train().to(device)
    trainloader = create_dataloader(trainData, augTrainLabels, batch_size)
    for epoch in tqdm(range(epochs)):
        for x, y in trainloader:
            # x = trainData[i]
            # y = augTrainLabels[i]
            # x = torch.from_numpy(x).float().to(device)
            # y = torch.from_numpy(y).float().to(device)
            x, y = x.to(device).float(), y.to(device).float()
            output = model(x)
            # print(output.shape, y.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    return model

def trainANN(model, name, trainpath, testpath, num_features, optimizer, criterion, epochs, batch_size, device):
    model.to(device)
    model.train()
    train_data, train_labels, test_data, test_labels = process_data(name, trainpath, testpath, num_features)
    train_dataloader = create_dataloader(train_data, train_labels, batch_size)

    for i in tqdm(range(epochs)):
        for x, y in train_dataloader:
            x, y = x.to(device).float(), y.to(device).float()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    metrics = Metrics(model, device, train_data, train_labels, test_data, test_labels)
    results = {}
    results['train'] = metrics.evaluate_on_whole_dataset(mode='train')
    results['test'] = metrics.evaluate_on_whole_dataset(mode='test')
    results['combined'] = metrics.evaluate_on_whole_dataset(mode='combined')
    return model, results

class Metrics:
    def __init__(self, model, device, train_data, train_labels, test_data, test_labels):
        self.model = model
        self.device = device
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.modes = ['train', 'test', 'combined']

    def evaluate_on_whole_dataset(self, mode='test'):
        print("Evaluating on {} set".format(mode))
        self.predict_on_train()
        self.predict_on_test()

        if mode == 'train':
            Y_true = self.train_labels
            Y_pred = self.train_pred_labels
        elif mode == 'test':
            Y_true = self.test_labels
            Y_pred = self.test_pred_labels
        else:
            Y_true = np.vstack([self.train_labels, self.test_labels])
            Y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])

        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  
                        'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 
                        'micro av. f1': [], 'macro av. f1': []}

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

    def continual_learning_evaluation(self, label_list, task_id=None, mode='test', continual_mode='individual'):
        print("Evaluating in {} mode on {} set".format(continual_mode, mode))
        self.predict_on_train()
        self.predict_on_test()
        cum_label_list = [0]
        for k in label_list:
            cum_label_list.append(cum_label_list[-1]+k)
        start_labels = cum_label_list[:-1]
        end_labels = cum_label_list[1:]
        
        if mode == 'train':
            y_true = self.train_labels
            y_pred = self.train_pred_labels
        elif mode == 'test':
            y_true = self.test_labels
            y_pred = self.test_pred_labels
        else:
            y_true = np.vstack([self.train_labels, self.test_labels])
            y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])
            
        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  
                        'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 
                        'micro av. f1': [], 'macro av. f1': []}
        
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
        # table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        # print(results_df.T.to_markdown())
        return results_dict
            
    def predict_on_train(self):
        self.model.eval()
        pred_labels = []
        for j in range(len(self.train_data)):
            x = torch.from_numpy(self.train_data[j]).to(self.device).float()
            yhat = self.model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        
        self.train_pred_labels = np.array(pred_labels)
        
    def predict_on_test(self):
        self.model.eval()
        pred_labels = []
        for j in range(len(self.test_data)):
            x = torch.from_numpy(self.test_data[j]).to(self.device).float()
            yhat = self.model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        
        self.test_pred_labels = np.array(pred_labels)



class NEWMetrics:
    def __init__(self, model, device, train_data, train_labels, test_data, test_labels):
        self.model = model
        self.device = device
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        # self.num_tasks = num_tasks
        self.modes = ['train', 'test', 'combined']

    def evaluate_on_whole_dataset(self, mode='test'):
        print("Evaluating on {} set".format(mode))
        self.predict_on_train()
        self.predict_on_test()

        if mode == 'train':
            Y_true = self.train_labels
            Y_pred = self.train_pred_labels
        elif mode == 'test':
            Y_true = self.test_labels
            Y_pred = self.test_pred_labels
        else:
            Y_true = np.vstack([self.train_labels, self.test_labels])
            Y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])

        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  
                        'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 
                        'micro av. f1': [], 'macro av. f1': [], 'imb. av. f1': []}

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
        results_dict['imb. av. f1'].append(self.imb_score(Y_true, Y_pred))
        results_df = pd.DataFrame.from_dict(results_dict)

        table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        print(results_df.T.to_markdown())
        return results_dict

    def continual_learning_evaluation(self, label_list, task_id=None, mode='test', continual_mode='individual'):
        print("Evaluating in {} mode on {} set".format(continual_mode, mode))
        self.predict_on_train()
        self.predict_on_test()
        cum_label_list = [0]
        for k in label_list:
            cum_label_list.append(cum_label_list[-1]+k)
        start_labels = cum_label_list[:-1]
        end_labels = cum_label_list[1:]

        num_tasks = len(label_list)
        
        if mode == 'train':
            y_true = self.train_labels
            y_pred = self.train_pred_labels
        elif mode == 'test':
            y_true = self.test_labels
            y_pred = self.test_pred_labels
        else:
            y_true = np.vstack([self.train_labels, self.test_labels])
            y_pred = np.vstack([self.train_pred_labels, self.test_pred_labels])
            
        results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  
                        'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 
                        'micro av. f1': [], 'macro av. f1': [], 'imb. av. f1': []}
        
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
            results_dict['imb. av. f1'].append(self.imb_score(Y_true, Y_pred))
            results_df = pd.DataFrame.from_dict(results_dict)

            # table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
            # print(results_df.T.to_markdown())
        else:
            for j in range(len(label_list)):
                if continual_mode == 'individual':
                    start_label = start_labels[j]
                else:
                    start_label = start_labels[0]
                end_label = end_labels[j]

                Y_true, Y_pred = y_true[:, start_label:end_label], y_pred[:, start_label:end_label]
    
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
                results_dict['imb. av. f1'].append(self.imb_score(Y_true, Y_pred))
                results_df = pd.DataFrame.from_dict(results_dict)
        
        # if task_id == num_tasks-1 or task_id is None:
            table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
            print(results_df.T.to_markdown())
        return results_dict
    
    def imb_score(self, y_true, y_pred):
        num_samples_per_label = y_true.sum(axis=0) + 1.0
        total_samples = num_samples_per_label.sum()
        weights = total_samples/num_samples_per_label
        f1_score_arr = []
        for label in range(y_true.shape[1]):
            f1_score_arr.append(f1_score(y_true[:, label], y_pred[:, label], average='micro')*weights[label])
        f1_score_arr = np.array(f1_score_arr)
        # prod = 1.0
        # for f1 in f1_score_arr:
        #     prod = prod*f1
        # f1_geom = prod ** (1.0/len(f1_score_arr))
        # f1_geom = gmean(f1_score_arr)
        f1_vals = f1_score_arr.sum()/(weights.sum())
        return f1_vals
            
    def predict_on_train(self):
        self.model.eval()
        pred_labels = []
        for j in range(len(self.train_data)):
            x = torch.from_numpy(self.train_data[j]).to(self.device).float()
            yhat = self.model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        
        self.train_pred_labels = np.array(pred_labels)
        
    def predict_on_test(self):
        self.model.eval()
        pred_labels = []
        for j in range(len(self.test_data)):
            x = torch.from_numpy(self.test_data[j]).to(self.device).float()
            yhat = self.model(x)
            pred_label = torch.zeros_like(yhat)
            pred_label[torch.where(yhat > 0)] = 1
            pred_labels.append(pred_label.detach().cpu().numpy())
        
        self.test_pred_labels = np.array(pred_labels)