import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from utils import *

def variable(t: torch.Tensor, device, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.to(device)
    else:
        t = t.to(torch.device('cpu'))
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model, trainData, trainLabels, device, criterion, optimizer, importance, batch_size):

        self.model = model
        self.data = trainData
        self.labels = trainLabels
        self.device = device
        self.criterion = criterion
        self.importance = importance
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.params = {n: p for n, p in self.model.feature_extractor.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        self.model.to(self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data, self.device)

    def change_model(self, model):
        self.model = model
        self.params = {n: p for n, p in self.model.feature_extractor.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data, self.device)
        
        self.model.to(self.device)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data, self.device)

        self.model.eval()
        trainloader = create_dataloader(self.data, self.labels, self.batch_size)
        for x, y in trainloader:
            # x = self.data[i]
            # y = self.labels[i]
            # x = torch.from_numpy(x).float().to(self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            self.model.zero_grad()
            x, y = x.to(self.device).float(), y.to(self.device).float()
            x = variable(x, self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            output = self.model(x)
            # label = torch.zeros_like(output)
            # label[torch.where(output) > 0] = 1
            loss = self.criterion(output, y)
            loss.backward()

            for n, p in self.model.feature_extractor.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.data)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.feature_extractor.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss = loss + _loss.sum()
        return loss * self.importance
    
    def change_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def train(self, model):
        model.train().to(self.device)
        trainloader = create_dataloader(self.data, self.labels, self.batch_size)
        for x, y in trainloader:
            # x = self.data[i]
            # y = self.labels[i]
            # x = torch.from_numpy(x).float().to(self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            x, y = x.to(self.device).float(), y.to(self.device).float()
            output = model(x)
            loss = self.criterion(output, y) + self.penalty(model)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return model

    
class SI(object):
    def __init__(self, model, trainData, trainLabels, device, criterion, optimizer, importance, batch_size):
        
        self.model = model
        self.data = trainData
        self.labels = trainLabels
        self.device = device
        self.criterion = criterion
        self.importance_factor = importance
        self.damping_factor = importance
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.params = {n: p for n, p in self.model.feature_extractor.named_parameters() if p.requires_grad}

        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

        self.importance = {}
        for n, p in self.params.items():
            self.importance[n] = p.clone().detach().fill_(0)

        self.model.to(self.device)

    def change_model(self, model):
        self.model = model
        self.params = {n: p for n, p in self.model.feature_extractor.named_parameters() if p.requires_grad}

        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

        self.importance = {}
        for n, p in self.params.items():
            self.importance[n] = p.clone().detach().fill_(0)

        self.model.to(self.device)

    def train(self, model):
        unreg_gradients = {}
        
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        loss = 0
        trainloader = create_dataloader(self.data, self.labels, self.batch_size)
        for x, y in trainloader:
            # x = self.data[i]
            # y = self.labels[i]
            # x = torch.from_numpy(x).float().to(self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            x, y = x.to(self.device).float(), y.to(self.device).float()
            x = variable(x, self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            output = self.model(x)
            loss_ = self.criterion(output, y)
            loss = loss + loss_
        
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        model = self.train_step(model, trainloader)

        for n, p in model.feature_extractor.named_parameters():
            # delta = p.detach() - old_params[n]
            delta = p.detach() - self.params[n].detach()
            if n in unreg_gradients.keys():  
                self.w[n] = self.w[n] - unreg_gradients[n] * delta  

        # for n, p in self.importance.items():
        #     delta_theta = self.params[n].detach() - old_params[n]
        #     p += self.w[n]/(delta_theta**2 + self.damping_factor)
        #     self.w[n].zero_()

        for n, p in model.feature_extractor.named_parameters():
            delta_theta = p.detach() - self.params[n].detach()
            p = p + self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()
        return model

    def penalty(self, model):
        loss = 0
        for (n, p), (r, q) in zip(model.feature_extractor.named_parameters(), self.model.feature_extractor.named_parameters()):
            # _loss = self.importance[n] * (p - self.model.feature_extractor.named_parameters()[n]) ** 2
            _loss = self.importance[n] * (p - q) ** 2
            loss = loss + _loss.sum()
        return loss * self.importance_factor

    def change_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def train_step(self, model, loader):
        model.train().to(self.device)
        # trainloader = create_dataloader(self.data, self.labels, self.batch_size)
        for x, y in loader:
            # x = self.data[i]
            # y = self.labels[i]
            # x = torch.from_numpy(x).float().to(self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            x, y = x.to(self.device).float(), y.to(self.device).float()
            output = model(x)
            loss = self.criterion(output, y) + self.penalty(model)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return model
    

class LwF(object):
    def __init__(self, model, trainData, trainLabels, device, optimizer, importance, batch_size):
        self.model = model
        self.data = trainData
        self.labels = trainLabels
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.importance_factor = importance
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.model.to(self.device)
        
    def change_model(self, model):
        self.model = model
        
    def change_optimizer(self, optimizer):
        self.optimizer = optimizer

    def penalty(self, model):
        model.train().to(self.device)
        loss = 0
        for i in range(len(self.data)):
            x = self.data[i]
            x = torch.from_numpy(x).float().to(self.device)
            output = model(x)[:self.model.out_dim]
            y = self.model(x).detach()
            loss_ = self.criterion(output, y)
            loss = loss + loss_

        return loss * self.importance_factor

    def train(self, model):
        model.train().to(self.device)
        trainloader = create_dataloader(self.data, self.labels, self.batch_size)
        for x, y in trainloader:
            # x = self.data[i]
            # y = self.labels[i]
            # x = torch.from_numpy(x).float().to(self.device)
            # y = torch.from_numpy(y).float().to(self.device)
            x, y = x.to(self.device).float(), y.to(self.device).float()
            output = model(x)
            loss = self.criterion(output, y) + self.penalty(model)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return model
