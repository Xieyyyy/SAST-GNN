import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import random


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype('float')
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = np.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = np.sqrt(mse)
    return mae, mape, rmse


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from os.path import join as pjoin
from data_utils import *
from utils.math_graph import *
from tqdm import tqdm
import os
import natsort
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# gpu_num = torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='METR-LA', help='')
parser.add_argument('--n_his', type=int, default=12, help='')
parser.add_argument('--n_pred', type=int, default=12, help='')
parser.add_argument('--hidden', type=int, default=16, help='')
parser.add_argument('--out_dim', type=int, default=2, help='')
parser.add_argument('--gnn', type=str, default='gcn', help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--test_size', type=int, default=4, help='')
parser.add_argument('--seed', type=int, default=123, help='')
parser.add_argument('--RESUME', type=bool, default=False, help='')
args = parser.parse_args()


# Set random seed
def set_random_seed(seed=10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


set_random_seed()

W =
