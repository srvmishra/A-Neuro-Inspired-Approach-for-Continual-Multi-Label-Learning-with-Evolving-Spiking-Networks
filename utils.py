import numpy as np
import pandas as pd
import markdown
from tqdm import tqdm

# def is_any_zero(data):
#     nlabels = data.shape[1]
#     for i in range(nlabels):
#         if (data[:, i] == 0).all():
#             return False
#     return True

def is_any_zero(data):
    # there is no random seed which does well
    # for any seed, there is at least one sample in at least one of the tasks which has all negative labels
    nsamples = data.shape[0]
    for i in range(nsamples):
        if (data[i, :] == 0).all():
            return False
    return True

# def compute_correlations(labels, labels_list, seed):
#     # corrmat = compute_label_correlation_matrix(labels)
#     # print("correlation matrix shape ", corrmat.shape)
#     compute_taskwise_correlations(labels, labels_list, seed)
#     return

def label_corr_table(labels, labels_list, seeds):
    col_labels = []
    for i in range(len(labels_list)):
        col_labels.append(str(i+1))
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            if j > i:
                col_labels.append(str(i+1) + '-' + str(j+1))
    res_dict = {}
    for seed in seeds:
        corrs = compute_taskwise_correlations(labels, labels_list, seed)
        res_dict["seed {}".format(seed)] = corrs
    results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=col_labels)
    print(results_df.to_markdown())
    return

def largest_indices(labels, n):
    """Returns the n largest indices from a numpy array."""
    corrmat = compute_label_correlation_matrix(labels)
    flat = corrmat.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, corrmat.shape)

def compute_label_correlation_matrix(labels):
    sum_ = lambda i, j: sum([1 for l in labels if l[i] == 1 and l[j] == 1])
    nlabels = labels.shape[1]
    corrmat = np.zeros((nlabels, nlabels))
    for i in range(nlabels):
        for j in range(nlabels):
            if i > j:
                s = sum_(j, i)
                corrmat[i, j] += s/labels.sum()
                corrmat[j, i] += s/labels.sum()
    return corrmat

def compute_taskwise_correlations(labels, labels_list, seed):
    cum_label_list = [0]
    for k in labels_list:
        cum_label_list.append(cum_label_list[-1]+k)
    start_labels = cum_label_list[:-1]
    end_labels = cum_label_list[1:]

    generator = np.random.RandomState(seed)
    indices = generator.permutation(labels.shape[1])
    labels_ = labels[:, indices]
    corrmat_ = compute_label_correlation_matrix(labels_)
    corr = lambda mat: mat.sum()/(mat.shape[0] * mat.shape[1])
    
    intra_ = []
    inter_ = []

    print("---------------------SEED {}---------------------".format(seed))
    for i in range(len(start_labels)):
        mat = corrmat_[start_labels[i]:end_labels[i], start_labels[i]:end_labels[i]]
        print("task id {} average intra label correlation {:.4f}".format(i+1, corr(mat)))
        intra_.append(corr(mat))

    for i in range(len(start_labels)):
        for j in range(len(start_labels)):
            if j > i:
                mat = corrmat_[start_labels[i]:end_labels[i], start_labels[j]:end_labels[j]]
                print("task id {} task id {} average inter label correlation {:.4f}".format(i+1, j+1, corr(mat)))
                inter_.append(corr(mat))
    return intra_ + inter_

def return_seed(samples_list, labels_list, train_dataset):
    # print("Train and Test labels match: {}".format(train_dataset.shape[1] == test_dataset.shape[1]))

    cum_sample_list = [0]
    for k in samples_list:
        cum_sample_list.append(cum_sample_list[-1]+k)
    start_samples = cum_sample_list[:-1]
    end_samples = cum_sample_list[1:]

    cum_label_list = [0]
    for k in labels_list:
        cum_label_list.append(cum_label_list[-1]+k)
    start_labels = cum_label_list[:-1]
    end_labels = cum_label_list[1:]

    seeds = []
    for seed in tqdm(range(3, 1000)):
        generator = np.random.RandomState(seed)
        indices = generator.permutation(train_dataset.shape[1])
        train_labels = train_dataset[:, indices]
        # test_labels = test_dataset[:, indices]
        n_zeros = 0
        for k in range(len(samples_list)):
            train_ = train_labels[start_samples[k]:end_samples[k], start_labels[k]:end_labels[k]]
            # test_ = test_labels[start_samples[k]:end_samples[k], start_labels[k]:end_labels[k]]
            if not is_any_zero(train_):
                n_zeros += 1
                break
        if n_zeros == 0:
            seeds.append(seed)
        if len(seeds) > 3:
            break
    return seeds