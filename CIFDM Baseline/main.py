import argparse
import copy
from time import time
import numpy as np
import torch
import pandas as pd
import markdown

from configuration import Config
from dataset import load_dataset
from model import OldFrontModel, OldEndModel, NewFrontModel, NewEndModel, IntermediaModel, AssistModel, ConcatOldModel, \
    ConcatNewModel, TeacherFrontModel, TeacherEndModel, ConcatTeacherModel, AssistEndModel, ConcatAssistModel
from other import AsymmetricLossOptimized
from tool import CorrelationMLSMLoss, init_weights, make_test, CorrelationMSELoss, TaskInfor, WeightCorrelationMSELoss, \
    CorrelationAsymmetricLoss
from train import train_single, train_joint, old_train_new, new_train_old_inter


def main(opt):
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    config = Config(opt)
    #device = torch.device('cuda:2')
    device = config.device
    print(config)
    train_list, test_train_list, test_data = load_dataset(config)
#     test_data = test_data[-1]
    
    print(test_data.data_y.shape)
    print(torch.sum(test_data.data_y).item()/(test_data.data_y.shape[0]*test_data.data_y.shape[1]))
    old_front_model = OldFrontModel(config.attri_num, config.embed_dim).to(device)
    old_end_model = OldEndModel(config.embed_dim, config.label_list[0]).to(device)
    new_front_model = NewFrontModel(config.attri_num, config.embed_dim).to(device)
    new_end_model = NewEndModel(config.embed_dim, config.label_list[1]).to(device)
    intermedia_model = IntermediaModel(config.embed_dim * 2, config.embed_dim).to(device)
    old_concate_model = ConcatOldModel(old_front_model, old_end_model).to(device)
    new_concate_model = ConcatNewModel(new_front_model, intermedia_model, new_end_model).to(device)
    
    for idx, train in enumerate(train_list):
        neg_count = 0
        pos_count = 0
        for _, y in train:
            if not y.any():
                neg_count += 1
            if y.all():
                pos_count += 1
        print("task: {}, #samples: {}, #negative samples: {}, #positive samples: {}".format(idx, len(train), neg_count, pos_count))

    # old_front_model.apply(init_weights)
    # old_end_model.apply(init_weights)
    # new_front_model.apply(init_weights)
    # new_end_model.apply(init_weights)
    # intermedia_model.apply(init_weights)
    # assist_model.apply(init_weights)
    
#     ind_results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': [], 'micro av. pr.': [], 'macro av. pr.': [], 'micro av. auc': [], 'macro av. auc': [], 'coverage error': [], 'ranking loss': []}
    
#     com_results_dict = {'hamming loss': [], 'zero_one_loss': [], 'one_error': [], 'micro av. jaccard': [], 'macro av. jaccard': [],  'micro av. precision': [], 'macro av. precision': [], 'micro av. recall': [], 'macro av. recall': [], 'micro av. f1': [], 'macro av. f1': [], 'micro av. pr.': [], 'macro av. pr.': [], 'micro av. auc': [], 'macro av. auc': [], 'coverage error': [], 'ranking loss': []}

    for i in range(config.task_num):
        print("======================== Task {} ========================".format(i))

        if i == 0:
            # Task 0, only train old model as base.
            for j in range(1):
                # print('The j is {}.'.format(j))
                temp_criterion = WeightCorrelationMSELoss(device, config.weight)#torch.nn.MSELoss()#
                temp_criterion = CorrelationMLSMLoss(device)
                temp_criterion = CorrelationAsymmetricLoss(device)
                train_single(old_concate_model, train_list[i], test_train_list[i], device, temp_criterion, config.first_batch, config.first_epoch, 16)
                print("The task {} result is following:".format(0))
                infor = TaskInfor([0], 'single')
                ind_results_dict = make_test(old_concate_model, new_concate_model, test_data, device, infor, config)

            old_train_new(old_concate_model, new_concate_model, train_list[i], device, config)
        else:
            # Other tasks need to adjust old model and train new model.
            train_joint(old_concate_model, new_concate_model, train_list[i], device, config)
            task_list = []

            for j in range(i+1):
                task_list.append(j)

            infor = TaskInfor(task_list, 'single')
            ind_results_dict = make_test(old_concate_model, new_concate_model, test_data, device, infor, config)

            if i != (config.task_num - 1):
                new_train_old_inter(old_concate_model, new_concate_model, train_list[i], device, config)
                # print('----------------------test teacher----------------------')
                # make_test_one(old_concate_model, test_data, device, infor, config)
                old_train_new(old_concate_model, new_concate_model, train_list[i], device, config)
        
        if i == config.task_num-1:
            print("Individual Task Results")
            results_df = pd.DataFrame.from_dict(ind_results_dict)
            print(results_df.T.to_markdown())

        print()

    print("======================== Final Result ========================")

    task_list = []
    for i in range(config.task_num):
        task_list.append(i)

    infor = TaskInfor(task_list, 'incremental')
    com_results_dict = make_test(old_concate_model, new_concate_model, test_data, device, infor, config)
    
    print("Overall Task Results")
    results_df = pd.DataFrame.from_dict(com_results_dict)
    print(results_df.T.to_markdown())

if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--log', type=str, default='/log/')
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    opt = parser.parse_args()
    torch.set_printoptions(profile="full")

    main(opt)
    print(time() - time_start, 's.')