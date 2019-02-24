import argparse
import os
import time

from omnidata import load_mnist
from omnimodel import Statistician
from omniplot import save_test_grid
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

import gzip
import numpy as np
import os
import pickle
import torch

from skimage.transform import rotate
from torch.utils import data

try:
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)
except ModuleNotFoundError:
    # put parent directory in path for utils
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)

filename = "../outputdilate4/checkpoints/15-02-2019-03:43:01-400.m"
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
model.eval()


def classify_datapoint(x, D_means, D_vars, K, single_sample=True):
    dataset = torch.from_numpy(x)
    with torch.no_grad():
        inputs = Variable(dataset.cuda())
    h1 = model.shared_convolutional_encoder(inputs)
    c_mean_, c_logvar_ = model.statistic_network(h1, summarize=False, single_sample=single_sample)
    kl_divergences = []
    for i in range(K):
        kl = kl_diagnormal_diagnormal(D_means[i], D_vars[i], c_mean_, c_logvar_)
        kl_divergences.append(kl.data.item())
    best_index = kl_divergences.index(min(kl_divergences))
    return best_index

def mnist_one_shot(K=10, support=1, n_trials=100, n_test_samples=1):
    W = support
    data_dir = "../mnist-data"
    images, one_hot_labels = load_mnist(data_dir=data_dir)
    lb = np.argmax(one_hot_labels, 1)
    for trial in tqdm(range(n_trials)):
        D = []
        x_test = []
        x_labels = []
        for i in range(K):
            idx = np.where(lb == i)[0]
            np.random.shuffle(idx)
            D.append(images[idx[:W]])
            samples = images[idx[W : W + n_test_samples]]
            x_test.append(samples)
            x_labels += ([i] * samples.shape[0])
        x_labels = np.array(x_labels)
        x_test = np.vstack(x_test)
        D = np.array(D)
        D_means = []
        D_vars = []
        for i in range(K):
            dataset = torch.from_numpy(D[i])
            with torch.no_grad():
                inputs = Variable(dataset.cuda())
            h = model.shared_convolutional_encoder(inputs)
            c_mean_full, c_logvar_full = model.statistic_network(h, summarize=True)
            D_means.append(c_mean_full)
            D_vars.append(c_logvar_full)
        test_loader = data.DataLoader(dataset=x_test, batch_size=100,
                                  shuffle=False, num_workers=0, drop_last=False)
        preds = []
        for batch in test_loader:
            with torch.no_grad():
                inputs = Variable(batch.cuda())
            h1 = model.shared_convolutional_encoder(inputs)
            c_mean_, c_logvar_ = model.statistic_network(h1, single_sample=True)
            for bi, x in enumerate(batch):
                kl_divergences = []
                for i in range(K):
                    kl = kl_diagnormal_diagnormal(D_means[i], D_vars[i], c_mean_[bi], c_logvar_[bi])
                    kl_divergences.append(kl.data.item())
                best_index = kl_divergences.index(min(kl_divergences))
                preds.append(best_index)
        acc = np.mean(np.array(preds) == np.array(x_labels))
        print(acc)
        accs.append(acc)
    return accs

accs_1 = mnist_one_shot(support=1, n_test_samples=1000, n_trials=100)
print("1-shot: ".format(np.mean(accs_1)))
accs_5 = mnist_one_shot(support=5, n_test_samples=1000, n_trials=100)
print("5-shot: ".format(np.mean(accs_5)))