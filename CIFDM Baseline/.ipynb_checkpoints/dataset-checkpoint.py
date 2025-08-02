import arff
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.linear_model import LogisticRegression
import scipy.io
from scipy.io import loadmat
import pandas as pd


class StreamDataset(Dataset):
    '''
    A standard dataset format for a given task.
    '''

    def __init__(self, data_x, data_y, task_id, transform=None, all_y=None):
        if isinstance(data_x, np.ndarray):
            self.data_x = torch.from_numpy(data_x).float()
        else:
            self.data_x = data_x
        if isinstance(data_y, np.ndarray):
            self.data_y = torch.from_numpy(data_y)
        else:
            self.data_y = data_y
        self.task_id = task_id
        self.label_num = data_y.shape[1]
        self.transform = transform
        self.all_y = all_y

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class ParallelDataset(Dataset):
    '''
    A dataset format that contains two inputs.
    '''

    def __init__(self, data_x, data_x2, data_y, task_id, transform=None):
        if isinstance(data_x, np.ndarray):
            self.data_x = torch.from_numpy(data_x).float()
        else:
            self.data_x = data_x
        if isinstance(data_y, np.ndarray):
            self.data_y = torch.from_numpy(data_y)
        else:
            self.data_y = data_y
        if isinstance(data_x2, np.ndarray):
            self.data_x2 = torch.from_numpy(data_x2).float()
        else:
            self.data_x2 = data_x2
        self.task_id = task_id
        self.label_num = data_y.shape[1]
        self.transform = transform

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x1 = self.data_x[idx]
        x2 = self.data_x2[idx]
        y = self.data_y[idx]

        if self.transform:
            x1, x2, y = self.transform(x1, x2, y)

        return x1, x2, y


class TestDataset(Dataset):
    '''
    A dataset format for test.
    '''

    def __init__(self, data_x, data_y, transform=None):
        if isinstance(data_x, np.ndarray):
            self.data_x = torch.from_numpy(data_x).float()
        else:
            self.data_x = data_x
        if isinstance(data_y, np.ndarray):
            self.data_y = torch.from_numpy(data_y)
        else:
            self.data_y = data_y
        self.label_num = self.data_y.shape[1]
        self.transform = transform

    def get_label_num(self):
        return self.label_num

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


class TestDatasetOld(Dataset):
    '''
    A dataset format for test.
    '''

    def __init__(self, data_x, data_y_dict, transform=None):
        self.data_x = data_x
        self.data_y_dict = data_y_dict
        self.task_id = -1
        self.data_y = data_y_dict[-1]
        self.label_num = self.data_y.shape[1]
        self.transform = transform

    def get_task_id(self):
        return self.task_id

    def get_label_num(self):
        return self.label_num

    def load_task(self, task_id):
        self.data_y = self.data_y_dict[task_id]
        self.label_num = self.data_y.shape[1]
        self.task_id = task_id

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def make_data_dict(data, attri_num, label_list, from_head):
    '''
    Assign labels to each task.
    :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param from_head: Whether the assign previous labels to the current task.
    :return: A dictionary that key is task id and returns specific labels for that task.
    '''
    data_dict = {}
    data_dict[-1] = data[:, :attri_num]  # -1 means it contains features rather than labels.
    rest_index = attri_num

    for i in range(len(label_list)):
        if from_head:
            data_dict[i] = data[:,
                           attri_num: rest_index + label_list[i]]  # Current task labels also contain previous labels.
        else:
            data_dict[i] = data[:, rest_index: rest_index + label_list[
                i]]  # Current task labels do not contain previous labels.

        rest_index += label_list[i]

    return data_dict


def make_train_dataset_list(data, attri_num, label_list, instance_list, from_head):
    '''
    To make a dataset to test training set.
    :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param instance_list: An array how many instances are assigned to each task.
    :param from_head: Whether the assign previous labels to the current task.
    :return: A dictionary that key is task id and returns specific instances with specific labels for that task.
    '''
    data_dict = make_data_dict(data, attri_num, label_list, from_head)
    data_list = []
    data_index = 0

    for i in range(len(label_list)):
        temp_data = StreamDataset(
            data_dict[-1][data_index: data_index + instance_list[i]],
            data_dict[i][data_index: data_index + instance_list[i]],
            i,
            None,
            data[data_index: data_index + instance_list[i], attri_num:]
        )
        data_list.append(temp_data)
        data_index += instance_list[i]

    return data_list


def make_test_dataset(data, attri_num, label_list):
    '''
    To make a testing dataset
     :param data: Original data.
    :param attri_num: The number of columns for attributes.
    :param label_list: An array shows how many labels are assigned to each task.
    :param instance_list: An array how many instances are assigned to each task.
    :return: A dictionary that key is task id and returns specific labels for that task.
    '''
    data_x = data[: , : attri_num]
    data_y = data[: , attri_num:]  # -1 means it contains all labels.
    data_test = TestDataset(data_x, data_y, None)

    return data_test


def load_dataset(config):
    '''
    Load dataset from file.
    :param data_name: The name of dataset.
    :param attri_num: The number of columns of attribute.
    :param label_list: An array shows how many labels are assigned to each task.
    :param train_instance_list: An array how many instances are assigned to each task.
    :param test_instance_list: An array how many instances are assigned to each task.
    :return: A tuple of lists. Each list contains a dataset for a task.
    '''
    if config.name == 'yeast':
        train_path = '../../datasets/yeast/yeast-train.arff'
        test_path = '../../datasets/yeast/yeast-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
#         train_instance_list = [500, 500, 500]
#         n_features = 103
#         labels_list = [7, 3, 3]
#         train_data, train_labels = train_data[:, :n_features], train_data[:, n_features:]
#         test_data, test_labels = test_data[:, :n_features], test_data[:, n_features:]
#         print("labels = ", sum(config.labels_list) == train_labels.shape[1] - 1)
#         print("train data = ", sum(config.train_instance_list) == train_labels.shape[0])
    elif config.name == 'emotions':
        train_path = '../../datasets/emotions/emotions-train.arff'
        test_path = '../../datasets/emotions/emotions-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
#         train_instance_list = [130, 130, 131]
#         n_features = 72
#         labels_list = [2, 2, 2]
#         train_data, train_labels = train_data[:, :n_features], train_data[:, n_features:]
#         test_data, test_labels = test_data[:, :n_features], test_data[:, n_features:]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'flags':
        train_path = '../../datasets/flags/flags-train.arff'
        test_path = '../../datasets/flags/flags-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
#         train_instance_list = [43, 43, 43]
#         n_features = 19
#         labels_list = [3, 2, 2]
#         train_data, train_labels = train_data[:, :n_features], train_data[:, n_features:]
#         test_data, test_labels = test_data[:, :n_features], test_data[:, n_features:]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'gpositive':
        train_path = '../../datasets/GpositivePseAAC/Gram_positivePseAAC519-train.mat'
        test_path = '../../datasets/GpositivePseAAC/Gram_positivePseAAC519-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [103, 103, 105]
#         n_features = 440
#         labels_list = [2, 1, 1]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'gnegative':
        train_path = '../../datasets/GnegativePseAAC/Gram_negativePseAAC1392-train.mat'
        test_path = '../../datasets/GnegativePseAAC/Gram_negativePseAAC1392-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [278, 280, 278]
#         n_features = 440
#         labels_list = [3, 2, 3]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'plant':
        train_path = '../../datasets/PlantPseAAC/PlantPseAAC978-train.mat'
        test_path = '../../datasets/PlantPseAAC/PlantPseAAC978-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [186, 186, 186]
#         n_features = 440
#         labels_list = [4, 4, 4]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'virus':
        train_path = '../../datasets/VirusPseAAC/VirusPseAAC207-train.mat'
        test_path = '../../datasets/VirusPseAAC/VirusPseAAC207-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [41, 41, 42]
#         n_features = 440
#         labels_list = [2, 2, 2]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'human':
        train_path = '../../datasets/HumanPseAAC/HumanPseAAC3106-train.mat'
        test_path = '../../datasets/HumanPseAAC/HumanPseAAC3106-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [621, 621, 622]
#         n_features = 440
#         labels_list = [4, 6, 4]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'eukaryote':
        train_path = '../../datasets/EukaryotePseAAC/EukaryotePseAAC7766-train.mat'
        test_path = '../../datasets/EukaryotePseAAC/EukaryotePseAAC7766-test.mat'
        train_ = loadmat(train_path)
        test_ = loadmat(test_path)
        train_data, train_labels = train_['data'], train_['labels']
        test_data, test_labels = test_['data'], test_['labels']
        train_data = np.hstack([train_data, train_labels])
        test_data = np.hstack([test_data, test_labels])
#         train_instance_list = [1552, 1552, 1554]
#         n_features = 440
#         labels_list = [8, 7, 7]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'scene':
        train_path = '../../datasets/scene/scene-train.arff'
        test_path = '../../datasets/scene/scene-test.arff'
        train_data = arff.load(open(train_path, 'rt'))
        train_data = np.array(train_data['data']).astype(np.float32)
        test_data = arff.load(open(test_path, 'rt'))
        test_data = np.array(test_data['data']).astype(np.float32)
#         train_instance_list = [405, 403, 403]
#         n_features = 294
#         labels_list = [2, 2, 2]
#         train_data, train_labels = train_data[:, :n_features], train_data[:, n_features:]
#         test_data, test_labels = test_data[:, :n_features], test_data[:, n_features:]
#         print("labels = ", sum(labels_list) == train_labels.shape[1])
#         print("train data = ", sum(train_instance_list) == train_labels.shape[0])
    elif config.name == 'foodtruck':
        train_path = '../../datasets/foodtruck/foodtruck-rand-hout-tra.arff'
        test_path = '../../datasets/foodtruck/foodtruck-rand-hout-tst.arff'
        num_features = 21

        train_data = arff.load(open(train_path, 'rt'))
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
        train_data = train_df.to_numpy()

        test_data = arff.load(open(test_path, 'rt'))
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
        test_data = test_df.to_numpy()
    else:
        print("The dataset {} is not supported.".format(config.data_name))
        return None

    total = config.attri_num
#     np.random.shuffle(train_data)

    for t in config.label_list:
        total += t

    if total > train_data.shape[1]:
        print(total)
        print("Error feature number.")
        return

    total = 0

    for t in config.train_instance_list:
        total += t

    if total > train_data.shape[0]:
        print(train_data.shape[0])
        print('Error train intance number.')
        return

    total = 0

    # for t in config.test_instance_list:
    #     total += t

    if total > train_data.shape[0]:
        print('Error test intance number.')
        return

    print(train_data.shape, test_data.shape)

    if config.shuffle:
        train_data, test_data = make_shuffle(train_data, test_data, config)

    train_list = make_train_dataset_list(train_data, config.attri_num, config.label_list, config.train_instance_list,
                                         False)
    test_train_list = make_train_dataset_list(train_data, config.attri_num, config.label_list,
                                              config.train_instance_list, True)
    test_list = make_test_dataset(test_data, config.attri_num, config.label_list)

    return train_list, test_train_list, test_list

def make_shuffle(train_data, test_data, config):
    shuffled_indices = config.generator.permutation(train_data.shape[0])
    train_data = train_data[shuffled_indices, :]
    temp_x = train_data[:, : config.attri_num]
    temp_y = train_data[:, config.attri_num:]
#     np.random.seed(94)
    shuffled_indices = config.generator.permutation(temp_y.shape[1])
    temp_y = temp_y[:, shuffled_indices]
    train_data = np.concatenate([temp_x, temp_y], 1)
    temp_x = test_data[:, : config.attri_num]
    temp_y = test_data[:, config.attri_num:]
    temp_y = temp_y[:, shuffled_indices]
    test_data = np.concatenate([temp_x, temp_y], 1)
    return train_data, test_data

def data_select_mask(data_y, ci0=0.01, ci1=0.1):
    np.set_printoptions(threshold=np.inf)
    round0 = np.logical_and(-ci0 < data_y, data_y < ci0)
    round1 = np.logical_and((1 - ci1) < data_y, data_y < (1 + ci1))
    psedu = np.logical_or(round0, round1).astype(int)
    return psedu


def data_select(data_x, data_y, select_num, config, ci0=0.1, ci1=0.3):
    '''
    Select useful and trustable instances for SSL.
    :param data_x: Features.
    :param data_y: Labels.
    :param select_num: Number of selected instance.
    :param confident: The probability that the instance has can be trusted.
    :return: A list of index of selected instances.
    '''
    in_data = []
    out_data = []

    for i in range(data_y.shape[0]):
        for y in data_y[i]:
            if (ci0 < y < (1 - ci1)) or (y < (-ci0)) or (y > (1 + ci1)):
                out_data.append(i)
                break
        else:  # this else is for break for
            # print(data_y[i].round(4))
            in_data.append(i)

    if (select_num == -1) or (len(in_data) == 0):  # Currently it will return here, ignore the rest part.
        print("There are {} instances that can be trusted.".format(len(in_data)))
        return np.array(in_data)

    elif len(out_data) == 0:
        shuffled_i = config.generator.permutation(len(in_data))[: select_num]
        return shuffled_i


def main():
    pass


if __name__ == '__main__':
    main()
