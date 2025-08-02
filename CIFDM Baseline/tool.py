from time import time
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, zero_one_loss, hamming_loss, jaccard_score, coverage_error, average_precision_score, label_ranking_loss
from torch.utils.data import DataLoader
import numpy as np
from dataset import StreamDataset, data_select_mask, ParallelDataset
import torch
import markdown

from other import AsymmetricLossOptimized


def init_weights(w, m='kaiming'):
    if m == 'kaiming':
        if type(w) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(w.weight)
    else:
        return

class TaskInfor:
    def __init__(self, task_list, method):
        self.task_list = task_list
        self.method = method

class IntervalLoss(torch.nn.Module):
    def __init__(self, loss_function):
        super(IntervalLoss, self).__init__()
        self.loss_function = loss_function()

    def forward(self, pred, label):
        loss = torch.zeros(pred.shape, device=pred.device)

        mask_round_zero = torch.logical_and(-0.2 < pred, pred < 0)
        mask_one = torch.logical_and(label == 1, mask_round_zero)
        loss[mask_one] = torch.masked_select(0.2 - (1 * pred), mask_one)

        mask_round_zero = torch.logical_and(0 <= pred, pred < 0.2)
        mask_one = torch.logical_and(label == 1, mask_round_zero)
        loss[mask_one] = torch.masked_select(0.2 - (1 * pred), mask_one)

        mask_round_one = torch.logical_and(0.8 < pred, pred < 1)
        mask_zero = torch.logical_and(label == 0, mask_round_one)
        loss[mask_zero] = torch.masked_select(0.2 * (pred - 1) + 1, mask_zero)

        mask_round_one = torch.logical_and(1 <= pred, pred < 1.2)
        mask_zero = torch.logical_and(label == 0, mask_round_one)
        loss[mask_zero] = torch.masked_select(0.2 - (1 * (pred - 1)), mask_zero)

        loss = loss.sum()
        loss += self.loss_function(pred, label)
        return loss

class HybridLoss(torch.nn.Module):
    def __init__(self, old_num, device, gamma=8, weight=1):
        super(HybridLoss, self).__init__()
        self.old_num = old_num
        self.mlsm = torch.nn.MultiLabelSoftMarginLoss()
        self.correlation = CorrelationLoss(device, weight)
        self.my_asy = AsyCrossEntropyLoss(device, gamma)
        self.other_asy = AsymmetricLossOptimized()

    def forward(self, pred, label):
        pred_old = pred[:, : self.old_num]
        pred_new = pred[:, self.old_num:]
        label_old = label[:, : self.old_num]
        label_new = label[:, self.old_num:]
        return self.my_asy(pred_old, label_old) + self.correlation(pred_new, label_new) + self.other_asy(pred_new, label_new) + self.correlation(pred_old, label_old)


class CorrelationMSELoss(torch.nn.Module):
    def __init__(self, device):
        super(CorrelationMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.correlation = CorrelationLoss(device)

    def forward(self, pred, label):
        return self.mse(pred, label) + self.correlation(pred, label)

class CorrelationAsymmetricLoss(torch.nn.Module):
    def __init__(self, device, gamma_pos=6, weight=1):
        super(CorrelationAsymmetricLoss, self).__init__()
        self.asy = AsymmetricLossOptimized(gamma_pos)
        self.correlation = CorrelationLoss(device, weight)
    def forward(self, pred, label):
        return self.asy(pred, label) + self.correlation(pred, label)

class WeightCorrelationMSELoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(WeightCorrelationMSELoss, self).__init__()

        if weight == 1:
            self.mse = torch.nn.MSELoss()
        else:
            self.mse = WeightMSELoss(device, weight)

        self.correlation = CorrelationLoss(device, weight)

    def forward(self, pred, label):
        return self.mse(pred, label) + self.correlation(pred, label)

class WeightMSELoss(torch.nn.Module):
    def __init__(self, device, weight=1):
        super(WeightMSELoss, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, pred, label):
        loss_matrix = (pred - label) ** 2
        loss_matrix *= ((self.weight - 1) * label) + 1
        return torch.sum(loss_matrix)

class WeightCrossEntropyLoss(torch.nn.Module):
    # Todo
    def __init__(self, device, weight=1):
        super(WeightCrossEntropyLoss, self).__init__()
        pass

class FocalLoss(torch.nn.Module):
    # Todo
    def __init__(self, device, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):

        pass

class AsyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, device, gamma=6, alpha=1):
        super(AsyCrossEntropyLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        return -torch.sum(torch.pow(label, self.gamma) * torch.log(pred)) * self.alpha


class CorrelationMLSMLoss(torch.nn.Module):
    def __init__(self, device):
        super(CorrelationMLSMLoss, self).__init__()
        self.mlsm = torch.nn.MultiLabelSoftMarginLoss()
        self.correlation = CorrelationLoss(device)

    def forward(self, pred, label):
        return self.mlsm(pred, label) + self.correlation(pred, label)


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

def produce_pseudo_data(data, model, device, method='mask'):
    model.eval()
    dataset = None
    data_y = torch.empty([0, model.get_out_dim()]).to(device)
    temp_loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=24)

    # Get the predictions of the old model to be the psudo labels.
    for x, _ in temp_loader:
        x = x.to(device)
        data_y = torch.cat([data_y, model(x)], 0)

    data_y = data_y.cpu().detach().numpy()

    if method == 'mask':
        mask = data_select_mask(data_y)
        dataset = ParallelDataset(data.data_x, mask, data_y.round(), data.task_id, None) if np.sum(mask) != 0 else None

        # preds = []
        # reals = []
        # counter = 0
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j] == 1:
        #             preds.append(data_y[i][j].round())
        #             temp = data.all_y[i]
        #             reals.append(temp[j])
        #             if temp[j] == 1:
        #                 counter+=1
        # print('mask', mask.shape, mask.shape[1]*mask.shape[0], np.sum(mask), counter/np.sum(mask), '%', "Acc", accuracy_score(np.array(reals), np.array(preds)), "Prec", precision_score(np.array(reals), np.array(preds)), "Recall", recall_score(np.array(reals), np.array(preds)))

    else:
        selected = data_select(data.data_x, data_y, -1)  # use inter or final to find suitable samples
        mask = np.ones((data.data_x.shape[0], model.get_out_dim()))

        # Fine tune the old model by psudo labels.
        if len(selected) != 0:
            # todo how about no data.
            selected_x = []
            selected_y = []
            selected_truth = []  # test selected performance

            for t in selected:
                selected_x.append(data.data_x[t])
                selected_y.append(data_y[t].round())
                selected_truth.append(data.all_y[t][: model.get_out_dim()]) # test selected performance

            dataset = ParallelDataset(np.array(selected_x), mask, np.array(selected_y), data.task_id, None)

            selected_y = np.array(selected_y) > 0.5 # test selected performance
            selected_truth = np.array(selected_truth) # test selected performance
            print(selected_y.shape, selected_truth.shape) # test selected performance
            print('None', data.data_x.shape[0], selected_y.shape[0], accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1)))
            # print("The selected accuracy is", accuracy_score(selected_truth, selected_y), accuracy_score(selected_truth.reshape(-1), selected_y.reshape(-1))) # test selected performance
    return dataset

# def test_individual_task(old_concate_model, new_concate_model, test_data, device, infor, config):
#     task_list = [0, 1, 2]
#     label_index = [0, 7, 10, 13]
#     test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
#     outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
#     groud_truth = np.empty((0, test_data.data_y.shape[1]))

#     old_front = old_concate_model.front.to(device).eval()
#     old_end = old_concate_model.end.to(device).eval()
#     new_front = new_concate_model.front.to(device).eval()
#     new_end = new_concate_model.end.to(device).eval()
#     inter = new_concate_model.inter.to(device).eval()

#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)
#         #x1 = new_front(x)
#         #x2 = old_front(x)
#         #pred1 = old_end(x1)
#         #pred2 = new_end(inter(x1, x2))
#         #pred = torch.cat([pred1, pred2], 1)
#         #outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
#         #pred2 = old_end(x2)
#         #outputs = np.concatenate([outputs, pred2.cpu().detach().numpy()], 0)
#         x1 = old_front(x)
#         x2 = new_front(x)
#         pred1 = old_end(x1)
#         pred2 = new_end(inter(x1, x2))
#         pred = torch.cat([pred1, pred2], 1)
#         outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
#         groud_truth = np.concatenate([groud_truth, y.cpu().detach().numpy()], 0)

#     print("output shape = ", outputs.shape)
#     print("ground truth shape = ", groud_truth.shape)

#     for idx in task_list:
#         if infor.method == 'single':
#             s_idx = label_index[idx]
#             e_idx = label_index[idx + 1]
#         elif infor.method == 'incremental':
#             s_idx = label_index[0]
#             e_idx = label_index[idx + 1]
#         else:
#             print("Error in the function make_test.")
#             exit(1)

#         if infor.method == 'single':
#             print("The task {} result is following:".format(idx))
#         elif infor.method == 'incremental':
#             print("Until the task {} result is following:".format(idx))

#         print("label index list", label_index)
#         print("s index", s_idx)
#         print("e index", e_idx)

#         real_label = groud_truth[:, s_idx: e_idx]
#         pred_label = outputs[:, s_idx: e_idx]

#         print("The test shape is {}.".format(real_label.shape))
#         # print("Test AUC: {}".format(roc_auc_score(real_label, pred_label, average='micro')))
#         np.set_printoptions(threshold=np.inf)
#         real_label = np.array(real_label) > 0.5
#         pred_label = np.array(pred_label) > 0.5
#         print("real label", real_label.shape)
#         print("pred_label", pred_label.shape)
#         print("after reshape")
#         print("real label", real_label.reshape(-1).shape)
#         print("pred_label", pred_label.reshape(-1).shape)
#         print("Test Accuracy: {}, {}".format(accuracy_score(real_label, pred_label),
#                                         accuracy_score(real_label.reshape(-1), pred_label.reshape(-1))))

#         print("Test micro-precision: {}".format(precision_score(real_label, pred_label, average='micro')))
#         print("Test micro-recall: {}".format(recall_score(real_label, pred_label, average='micro')))
#         print("Test micro-F1: {}".format(f1_score(real_label, pred_label, average='micro')))

#         #print("Test micro-precision: {}".format(precision_score(real_label, pred_label)))
#         #print("Test micro-recall: {}".format(recall_score(real_label, pred_label)))
#         #print("Test micro-F1: {}".format(f1_score(real_label, pred_label)))

#         # print("Test macro-precision: {}".format(precision_score(real_label, pred_label, average='macro')))
#         # print("Test macro-recall: {}".format(recall_score(real_label, pred_label, average='macro')))
#         # print("Test macro-F1: {}".format(f1_score(real_label, pred_label, average='macro')))
#         print()

# def test_data_stats(old_concate_model, new_concate_model, test_data, device, infor, config):
#     task_list = [0, 1, 2]
#     label_index = [0, 7, 10, 13]
#     #test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
#     outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
#     groud_truth = np.empty((0, test_data.data_y.shape[1]))

#     old_front = old_concate_model.front.to(device).eval()
#     old_end = old_concate_model.end.to(device).eval()
#     new_front = new_concate_model.front.to(device).eval()
#     new_end = new_concate_model.end.to(device).eval()
#     inter = new_concate_model.inter.to(device).eval()

#     summary_dict = {'sample_id': [], 'ground_truth': [], 'pred_probs': [], 'pred_label': [], 'task_id': []}

#     for i, (x, y) in enumerate(test_data):
#         x = torch.from_numpy(x).unsqueeze(0).to(device)
#         #y = y.numpy()
#         for task_id in task_list:
#             s_id = label_index[task_id]
#             e_id = label_index[task_id + 1]            
#             if (y[s_id:e_id] == 0).all():
#                 x1 = old_front(x)
#                 x2 = new_front(x)
#                 pred1 = old_end(x1)
#                 pred2 = new_end(inter(x1, x2))
#                 pred = torch.cat([pred1, pred2], 1).detach().numpy()[0]
#                 summary_dict['sample_id'].append(i)
#                 summary_dict['ground_truth'].append(y[s_id:e_id])
#                 summary_dict['pred_probs'].append(pred[s_id:e_id])
#                 summary_dict['pred_label'].append((pred[s_id:e_id]>0.5).astype(np.float32))
#                 summary_dict['task_id'].append(task_id)
    
#     df = pd.DataFrame(summary_dict)
#     print(tabulate(df, tablefmt="pipe", headers="keys"))
#     return

# def new_test_data_stats(old_concate_model, new_concate_model, test_data, device, infor, config, task_id):
#     task_list = [0, 1, 2]
#     label_index = [0, 7, 10, 13]
#     s_id = label_index[0]
#     e_id = label_index[task_id + 1] 
#     #test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
#     outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
#     groud_truth = np.empty((0, test_data.data_y.shape[1]))

#     old_front = old_concate_model.front.to(device).eval()
#     old_end = old_concate_model.end.to(device).eval()
#     new_front = new_concate_model.front.to(device).eval()
#     new_end = new_concate_model.end.to(device).eval()
#     inter = new_concate_model.inter.to(device).eval()

#     summary_dict = {'sample_id': [], 'ground_truth': [], 'pred_probs': [], 'pred_label': [], 'task_id': []}

#     for i, (x, y) in enumerate(test_data):
#         x = torch.from_numpy(x).unsqueeze(0).to(device)
#         #y = y.numpy()
#         #for task_id in task_list:
#         #    s_id = label_index[task_id]
#         #    e_id = label_index[task_id + 1]            
#         if (y[s_id:e_id] == 0).all():
#             x1 = old_front(x)
#             x2 = new_front(x)
#             pred1 = old_end(x1)
#             pred2 = new_end(inter(x1, x2))
#             pred = torch.cat([pred1, pred2], 1).detach().numpy()[0]
#             summary_dict['sample_id'].append(i)
#             summary_dict['ground_truth'].append(y[s_id:e_id])
#             summary_dict['pred_probs'].append(pred[s_id:e_id])
#             summary_dict['pred_label'].append((pred[s_id:e_id]>0.5).astype(np.float32))
#             summary_dict['task_id'].append(task_id)
    
#     df = pd.DataFrame(summary_dict)
#     print(tabulate(df, tablefmt="pipe", headers="keys"))
#     return

# def make_new_test(old_concate_model, new_concate_model, test_data, device, infor, config, task_id):
#     # todo check details and modify usage
#     label_index = [0]
#     for l in config.label_list:
#         label_index.append(l+label_index[-1])

#     test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
#     outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
#     groud_truth = np.empty((0, test_data.data_y.shape[1]))

#     old_front = old_concate_model.front.to(device).eval()
#     old_end = old_concate_model.end.to(device).eval()
#     new_front = new_concate_model.front.to(device).eval()
#     new_end = new_concate_model.end.to(device).eval()
#     inter = new_concate_model.inter.to(device).eval()

#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)
#         #x1 = new_front(x)
#         #x2 = old_front(x)
#         #pred1 = old_end(x1)
#         #pred2 = new_end(inter(x1, x2))
#         #pred = torch.cat([pred1, pred2], 1)
#         #outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
#         #pred2 = old_end(x2)
#         #outputs = np.concatenate([outputs, pred2.cpu().detach().numpy()], 0)
#         x1 = old_front(x)
#         x2 = new_front(x)
#         pred1 = old_end(x1)
#         pred2 = new_end(inter(x1, x2))
#         pred = torch.cat([pred1, pred2], 1)
#         outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
#         groud_truth = np.concatenate([groud_truth, y.cpu().detach().numpy()], 0)

#     print("output shape = ", outputs.shape)
#     print("ground truth shape = ", groud_truth.shape)
    
#     real_label = groud_truth
#     pred_label = outputs[:, label_index[0]:label_index[task_id+1]]

#     print("The test shape is {}.".format(real_label.shape))
#     # print("Test AUC: {}".format(roc_auc_score(real_label, pred_label, average='micro')))
#     np.set_printoptions(threshold=np.inf)
#     real_label = np.array(real_label) > 0.5
#     pred_label = np.array(pred_label) > 0.5
#     print("real label", real_label.shape)
#     print("pred_label", pred_label.shape)
#     print("after reshape")
#     print("real label", real_label.reshape(-1).shape)
#     print("pred_label", pred_label.reshape(-1).shape)
#     print("Test Accuracy: {}, {}".format(accuracy_score(real_label, pred_label),
#                                   accuracy_score(real_label.reshape(-1), pred_label.reshape(-1))))

#     print("Test micro-precision: {}".format(precision_score(real_label, pred_label, average='micro')))
#     print("Test micro-recall: {}".format(recall_score(real_label, pred_label, average='micro')))
#     print("Test micro-F1: {}".format(f1_score(real_label, pred_label, average='micro')))

#     #print("Test micro-precision: {}".format(precision_score(real_label, pred_label)))
#     #print("Test micro-recall: {}".format(recall_score(real_label, pred_label)))
#     #print("Test micro-F1: {}".format(f1_score(real_label, pred_label)))

#     # print("Test macro-precision: {}".format(precision_score(real_label, pred_label, average='macro')))
#     # print("Test macro-recall: {}".format(recall_score(real_label, pred_label, average='macro')))
#     # print("Test macro-F1: {}".format(f1_score(real_label, pred_label, average='macro')))
#     print()

def one_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    err = 0
    for i in range(y_true.shape[0]):
        err = err + (y_true[i, :] != y_pred[i, :]).all()
    val = err/y_true.shape[0]
    return val

def imbalance_averaged_f1_score(y_true, y_pred):
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

def make_test(old_concate_model, new_concate_model, test_data, device, infor, config):
    # todo check details and modify usage
    label_index = [0]
    for l in config.label_list:
        label_index.append(l+label_index[-1])

    test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
    outputs = np.empty((0, old_concate_model.get_out_dim()+new_concate_model.get_out_dim()))
    groud_truth = np.empty((0, test_data.data_y.shape[1]))

    old_front = old_concate_model.front.to(device).eval()
    old_end = old_concate_model.end.to(device).eval()
    new_front = new_concate_model.front.to(device).eval()
    new_end = new_concate_model.end.to(device).eval()
    inter = new_concate_model.inter.to(device).eval()

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        #x1 = new_front(x)
        #x2 = old_front(x)
        #pred1 = old_end(x1)
        #pred2 = new_end(inter(x1, x2))
        #pred = torch.cat([pred1, pred2], 1)
        #outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
        #pred2 = old_end(x2)
        #outputs = np.concatenate([outputs, pred2.cpu().detach().numpy()], 0)
        x1 = old_front(x)
        x2 = new_front(x)
        pred1 = old_end(x1)
        pred2 = new_end(inter(x1, x2))
        pred = torch.cat([pred1, pred2], 1)
        outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
        groud_truth = np.concatenate([groud_truth, y.cpu().detach().numpy()], 0)

    print("output shape = ", outputs.shape)
    print("ground truth shape = ", groud_truth.shape)
    
    print("Evaluation mode is {}".format(infor.method))
    # results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': [], 'micro av. pr.': [], 'macro av. pr.': [], 'micro av. auc': [], 'macro av. auc': [], 'coverage error': [], 'ranking loss': []}
    results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': [], 'imb. av. f1': []}

#     results_dict = {'micro av. jaccard': [], 'micro av. f1': []}
    # results_dict = {'macro av. precision': [], 'macro av. recall': [], 'macro av. f1': [], 'macro av. pr.': [], 'macro av. auc': []}
    # results_dict = {'micro av. jaccard': [], 'micro av. precision': [], 'micro av. recall': [], 'micro av. f1': []}
    for idx in infor.task_list:
        if infor.method == 'single':
            s_idx = label_index[idx]
            e_idx = label_index[idx + 1]
        elif infor.method == 'incremental':
            s_idx = label_index[0]
            e_idx = label_index[idx + 1]
        else:
            print("Error in the function make_test.")
            exit(1)

#         if infor.method == 'single':
#             print("The task {} result is following:".format(idx))
#         elif infor.method == 'incremental':
#             print("Until the task {} result is following:".format(idx))

        print("label index list", label_index)
#         print("Task", idx)
#         print("s index", s_idx)
#         print("e index", e_idx)
        print("Task: {}, s index: {}, e index: {}".format(idx, s_idx, e_idx))

        Y_true = groud_truth[:, s_idx: e_idx]
        Y_score = outputs[:, s_idx: e_idx]

        print("The test shape is {}.".format(groud_truth.shape))
        # print("Test AUC: {}".format(roc_auc_score(real_label, pred_label, average='micro')))
        np.set_printoptions(threshold=np.inf)
        Y_true = np.array(Y_true) > 0.5
        Y_pred = np.array(Y_score) > 0.5
        
        results_dict['hamming loss'].append(hamming_loss(Y_true, Y_pred))
        results_dict['zero_one_loss'].append(zero_one_loss(Y_true, Y_pred))
        results_dict['one_error'].append(one_error(Y_true, Y_pred))
        results_dict['micro av. jaccard'].append(jaccard_score(Y_true, Y_pred, average='micro'))
        results_dict['macro av. jaccard'].append(jaccard_score(Y_true, Y_pred, average='macro'))
        results_dict['micro av. precision'].append(precision_score(Y_true, Y_pred, average='micro', zero_division=0))
        results_dict['macro av. precision'].append(precision_score(Y_true, Y_pred, average='macro', zero_division=0))
        results_dict['micro av. recall'].append(recall_score(Y_true, Y_pred, average='micro', zero_division=0))
        results_dict['macro av. recall'].append(recall_score(Y_true, Y_pred, average='macro', zero_division=0))
        results_dict['micro av. f1'].append(f1_score(Y_true, Y_pred, average='micro', zero_division=0))
        results_dict['macro av. f1'].append(f1_score(Y_true, Y_pred, average='macro', zero_division=0))
        results_dict['imb. av. f1'].append(imbalance_averaged_f1_score(Y_true, Y_pred))
        # results_dict['micro av. pr.'].append(average_precision_score(Y_true, Y_score, average='micro'))
        # results_dict['macro av. pr.'].append(average_precision_score(Y_true, Y_score, average='macro'))
        # results_dict['micro av. auc'].append(roc_auc_score(Y_true, Y_score, average='micro'))
        # results_dict['macro av. auc'].append(roc_auc_score(Y_true, Y_score, average='macro'))
        # results_dict['coverage error'].append(coverage_error(Y_true, Y_score))
        # results_dict['ranking loss'].append(label_ranking_loss(Y_true, Y_score))
        results_df = pd.DataFrame.from_dict(results_dict)
            
        table_html=markdown.markdown(results_df.T.to_markdown(), extensions=['markdown.extensions.tables'])
        # print(results_df.T.to_markdown())
        
#         print("real label", real_label.shape)
#         print("pred_label", pred_label.shape)
#         print("after reshape")
#         print("real label", real_label.reshape(-1).shape)
#         print("pred_label", pred_label.reshape(-1).shape)
#         print("Test Accuracy: {}, {}".format(accuracy_score(real_label, pred_label),
#                                         accuracy_score(real_label.reshape(-1), pred_label.reshape(-1))))

#         print("Test micro-precision: {}".format(precision_score(real_label, pred_label, average='micro')))
#         print("Test micro-recall: {}".format(recall_score(real_label, pred_label, average='micro')))
#         print("Test micro-F1: {}".format(f1_score(real_label, pred_label, average='micro')))

        #print("Test micro-precision: {}".format(precision_score(real_label, pred_label)))
        #print("Test micro-recall: {}".format(recall_score(real_label, pred_label)))
        #print("Test micro-F1: {}".format(f1_score(real_label, pred_label)))

        # print("Test macro-precision: {}".format(precision_score(real_label, pred_label, average='macro')))
        # print("Test macro-recall: {}".format(recall_score(real_label, pred_label, average='macro')))
        # print("Test macro-F1: {}".format(f1_score(real_label, pred_label, average='macro')))
        print()
    return results_dict

# def make_test_one(concate_model, test_data, device, infor, config):
#     # todo check details and modify usage
#     label_index = [0]
#     for l in config.label_list:
#         label_index.append(l+label_index[-1])

#     test_loader = DataLoader(test_data, batch_size=config.eval_batch, shuffle=False, num_workers=4)
#     concate_model.to(device).eval()

#     outputs = np.empty((0, concate_model.get_out_dim()))
#     groud_truth = np.empty((0, test_data.data_y.shape[1]))

#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)
#         pred = concate_model(x)
#         outputs = np.concatenate([outputs, pred.cpu().detach().numpy()], 0)
#         groud_truth = np.concatenate([groud_truth, y.cpu().detach().numpy()], 0)

#     for idx in infor.task_list:
#         if infor.method == 'single':
#             s_idx = label_index[idx]
#             e_idx = label_index[idx + 1]
#         elif infor.method == 'incremental':
#             s_idx = label_index[0]
#             e_idx = label_index[idx + 1]
#         else:
#             print("Error in the function make_test.")
#             exit(1)

#         if infor.method == 'single':
#             print("The task {} result is following:".format(idx))
#         elif infor.method == 'incremental':
#             print("Until the task {} result is following:".format(idx))

#         real_label = groud_truth[:, s_idx: e_idx]
#         pred_label = outputs[:, s_idx: e_idx]

#         print("The test shape is {}.".format(real_label.shape))
#         # print("Test AUC: {}".format(roc_auc_score(real_label, pred_label, average='micro')))
#         np.set_printoptions(threshold=np.inf)

#         # print(real_label[8:12])
#         # print(pred_label[8:12])
#         real_label = np.array(real_label) > 0.5
#         pred_label = np.array(pred_label) > 0.5
#         print("Test Accuracy: {}, {}".format(accuracy_score(real_label, pred_label),
#                                         accuracy_score(real_label.reshape(-1), pred_label.reshape(-1))))
#         # print("Test AUC: {}".format(roc_auc_score(real_label.reshape(-1), pred_label.reshape(-1))))
#         print("Test Precision: {}".format(precision_score(real_label.reshape(-1), pred_label.reshape(-1))))
#         print("Test Recall: {}".format(recall_score(real_label.reshape(-1), pred_label.reshape(-1))))
#         print()

def main():
    lossfunc = CorrelationLoss(torch.device('cpu'))
    loss = lossfunc(
        torch.Tensor([[0.1, 0.9, 0.3], [0.13, 0.87, 0.31]]),
        torch.Tensor([[0, 0, 0], [1, 0, 1]])
    )
    lossfunc = WeightMSELoss(torch.device('cpu'))
    loss = lossfunc(
        torch.Tensor([[0.1, 0.9, 0.3], [0.13, 0.87, 0.31]]),
        torch.Tensor([[0, 0, 0], [1, 0, 1]])
    )
    return


if __name__ == '__main__':
    main()
