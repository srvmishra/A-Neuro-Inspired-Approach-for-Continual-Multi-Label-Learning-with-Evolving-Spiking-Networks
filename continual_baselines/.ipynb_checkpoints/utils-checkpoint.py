import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, zero_one_loss, hamming_loss, jaccard_score
from sklearn.utils.multiclass import type_of_target
import torch
from random import shuffle
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
        super(self, MLP).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU,
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

class Metrics:
    def __init__(self, model, device, train_data, train_labels, test_data, test_labels):
        self.model = model
        self.device = device
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.modes = ['train', 'test', 'combined']
            
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
