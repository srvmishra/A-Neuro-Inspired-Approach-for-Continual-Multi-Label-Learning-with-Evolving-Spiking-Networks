import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from MultiLabelEvolvingSNN import *
import arff
import warnings
from scipy.io import loadmat
from utils import *
warnings.filterwarnings("ignore")

np.random.seed(0)
gen = np.random.RandomState(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compress(X, y):
    # lda = LinearDiscriminantAnalysis(n_components=2)
    # X_lda = lda.fit_transform(X, y)  # Reduce dimensions to 2
    # X_lda_class1 = X_lda[y == 1]
    # X_lda_class0 = X_lda[y == 0]
    # return X_lda_class1, X_lda_class0

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca_class1 = X_pca[y == 1]
    X_pca_class0 = X_pca[y == 0]
    return X_pca_class1, X_pca_class0, pca
    

def cluster(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=gen).fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centers, labels


def find_weights(net):
    class1_pos = [i for i in range(len(net.nature)) if net.nature[i] == 1]
    class0_pos = [i for i in range(len(net.nature)) if net.nature[i] == 0]
    class1_weights = net.weights[class1_pos, :]
    class0_weights = net.weights[class0_pos, :]

    weights_data_ = torch.vstack([class1_weights, class0_weights]).cpu().numpy()
    weights_labels_ = np.array([1.]*len(class1_pos) + [0.]*len(class0_pos))
    return weights_data_, weights_labels_, len(class1_pos), len(class0_pos)

def get_unique_legends(axs):
    legend_entries = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in legend_entries:  # Store unique entries only
                legend_entries[l] = h
    return legend_entries

def plot_clusters(axs, class_data_, class_centers_, extra_points_, titles_, cluster_labels_):
    class_1_ax, class_0_ax = axs
    class_1_data, class_0_data = class_data_
    class_1_centers, class_0_centers = class_centers_
    extra_class_1_points,  extra_class_0_points = extra_points_
    class_1_titles, class_0_titles = titles_
    class_1_cluster_labels_, class_0_cluster_labels_ = cluster_labels_

    class_1_ax.scatter(class_1_data[:, 0], class_1_data[:, 1], c=class_1_cluster_labels_, linewidth=1.5, cmap='viridis', alpha=0.6)
    class_1_ax.scatter(class_1_centers[:, 0], class_1_centers[:, 1], color='black', marker='+', linewidth=2,
                       s=150, label="Class 1 Cluster Centers")
    class_1_ax.scatter(extra_class_1_points[:, 0], extra_class_1_points[:, 1], color='black', marker='o', s=125, label=r"$S^+$ weights")
    class_1_ax.set_title(class_1_titles, fontsize=16)
    # class_1_ax.legend()

    class_0_ax.scatter(class_0_data[:, 0], class_0_data[:, 1], c=class_0_cluster_labels_, linewidth=1.5, cmap='plasma', alpha=0.6)
    class_0_ax.scatter(class_0_centers[:, 0], class_0_centers[:, 1], color='black', marker='x', linewidth=2,
                       s=150, label="Class 0 Cluster Centers")
    class_0_ax.scatter(extra_class_0_points[:, 0], extra_class_0_points[:, 1], color='black', marker='^', s=125, label=r"$S^-$ weights")
    class_0_ax.set_title(class_0_titles, fontsize=16)
    # class_0_ax.legend()
    return

def print_neurons(hparams):
    overall_net = OverallSNN(hparams)

    for id, net in enumerate(overall_net.nets):
        class1_pos = [i for i in range(len(net.nature)) if net.nature[i] == 1]
        class0_pos = [i for i in range(len(net.nature)) if net.nature[i] == 0]

        print("Label {}, S+ interneurons = {}, S- interneurons = {}".format(id, len(class1_pos), len(class0_pos)))
    return overall_net

def plot_spike_clusters(dataset_name, spike_data, labels, net, labels_list):
    # overall_net = print_neurons(hparams)
    overall_net = net

    # convert spike data to norm srf and take the third time
    # srf = np.expand_dims(spike_data, axis=1).transpose(1, 2) - overall_net.times.cpu().numpy()
    srf = np.transpose(np.expand_dims(spike_data, axis=1), [0, 2, 1]) - overall_net.times.cpu().numpy()
    # print(srf.shape)
    norm_srf = srf/np.expand_dims(srf.sum(1), axis=1)
    # print(norm_srf.shape)
    spike_data_ = norm_srf[:, :, overall_net.third_time]
    assert spike_data_.shape == spike_data.shape
    # use the third time for your calculations of cluster centers and other things

    fig, axs = plt.subplots(2, len(labels_list), figsize=(5 * len(labels_list), 2 * 5))
    plt.tight_layout()

    for id, l in enumerate(labels_list):
        net_ = overall_net.nets[l]
        weights_, weights_labels_, num_class_1, num_class_0 = find_weights(net_)

        compressed_data_class_1, compressed_data_class_0, pca = compress(spike_data_, labels[:, l])
        compressed_weights_class_1 = pca.transform(weights_[weights_labels_ == 1])
        compressed_weights_class_0 = pca.transform(weights_[weights_labels_ == 0])
        # compressed_weights_class_1, compressed_weights_class_0 = compress(weights_, weights_labels_)

        clustered_data_class_1_centers, class_1_data_labels = cluster(compressed_data_class_1, num_class_1)
        clustered_data_class_0_centers, class_0_data_labels = cluster(compressed_data_class_0, num_class_0)

        class_1_title = '$|S^+_{{}}| = {}$ clusters for $y_{{}} = 1$'.format(l, num_class_1, l)
        class_0_title = '$|S^-_{{}}| = {}$ clusters for $y_{{}} = 0$'.format(l, num_class_0, l)

        axs_ = [axs[0, id], axs[1, id]]
        class_data_ = [compressed_data_class_1, compressed_data_class_0]
        class_centers_ = [clustered_data_class_1_centers, clustered_data_class_0_centers]
        extra_points_ = [compressed_weights_class_1, compressed_weights_class_0]
        titles_ = [class_1_title, class_0_title]
        cluster_labels_ = [class_1_data_labels, class_0_data_labels]

        plot_clusters(axs_, class_data_, class_centers_, extra_points_, titles_, cluster_labels_)

    legend_entries = get_unique_legends(axs.flatten())
    fig.legend(legend_entries.values(), legend_entries.keys(), loc="lower center", ncol=len(legend_entries), 
               bbox_to_anchor=(0.5, -0.1), fontsize=16)
    
    plt.suptitle(dataset_name, fontsize=20)
    plt.subplots_adjust(top=0.9, hspace=0.25, wspace=0.25)
    plt.show()
    return