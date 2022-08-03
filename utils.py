from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, CoraFull
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F

import os.path as osp
import os

import numpy as np
# np.random.seed(0)

import torch
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from datetime import datetime

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Augmentation:

    def __init__(self, p_f1=0.2, p_f2=0.1, p_e1=0.2, p_e2=0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"

    def _feature_masking(self, data, device):
        feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        x1, x2 = data.x.clone(), data.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.x, new_data2.x = x1, x2
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr, new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def __call__(self, data):
        return self._feature_masking(data)


def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "cs" :
        dataset = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "physics":
        dataset = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "wikics":
        dataset = "WikiCS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": WikiCS, "src": "pyg"}
    elif dataset == "corafull":
        dataset = "CoraFull"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": CoraFull, "src": "pyg"}

    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data, name, public = None):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if public != None :
        train_idx = public["train"].numpy()
        valid_idx = public["valid"].numpy()
        test_idx = public["test"].numpy()

        _train_mask = _val_mask = _test_mask = None

        for i in range(20):
            data_index = np.arange(data.y.shape[0])
            train_mask = torch.tensor(np.in1d(data_index, train_idx), dtype=torch.bool)
            test_mask = torch.tensor(np.in1d(data_index, test_idx), dtype=torch.bool)
            valid_mask = torch.tensor(np.in1d(data_index, valid_idx), dtype=torch.bool)

            test_mask = test_mask.reshape(1, -1)
            valid_mask = valid_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if _train_mask is None:
                _train_mask = train_mask
                _val_mask = valid_mask
                _test_mask = test_mask

            else:
                _train_mask = torch.cat((_train_mask, train_mask), dim=0)
                _val_mask = torch.cat((_val_mask, valid_mask), dim=0)
                _test_mask = torch.cat((_test_mask, test_mask), dim=0)

        data.train_mask = _train_mask
        data.val_mask = _val_mask
        data.test_mask = _test_mask

    elif name != "WikiCS":

        _train_mask = _val_mask = _test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if _train_mask is None:
                _train_mask = train_mask
                _val_mask = dev_mask
                _test_mask = test_mask

            else:
                _train_mask = torch.cat((_train_mask, train_mask), dim=0)
                _val_mask = torch.cat((_val_mask, dev_mask), dim=0)
                _test_mask = torch.cat((_test_mask, test_mask), dim=0)
        
        data.train_mask = _train_mask
        data.val_mask = _val_mask
        data.test_mask = _test_mask

    else:  # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        self.step += 1
        return old * self.beta + (1 - self.beta) * new


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class KLD(nn.Module):
    def forward(self, inputs, targets):

        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        
        return F.kl_div(inputs, targets, reduction='batchmean')
