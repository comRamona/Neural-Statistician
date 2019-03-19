import argparse
import os
import time
import numpy as np
from copy import deepcopy
from omnidata import OmniglotSetsDataset, load_mnist_test_batch
from omnimodel import Statistician
from omniplot import save_test_grid
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
import torch



import logging
logging.basicConfig(filename='examplebin.log',level=logging.DEBUG)

# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')
parser.add_argument('--mnist-data-dir', required=True, type=str, default=None,
                    help='location of MNIST data (required for few shot learning)')
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size (of datasets) for training (default: 32)')
parser.add_argument('--sample-size', type=int, default=5,
                    help='number of samples per dataset (default: 5)')
parser.add_argument('--c-dim', type=int, default=512,
                    help='dimension of c variables (default: 512)')
parser.add_argument('--n-hidden-statistic', type=int, default=3,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=256,
                    help='dimension of hidden layers in statistic network (default: 256)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=16,
                    help='dimension of z variables (default: 16)')
parser.add_argument('--n-hidden', type=int, default=3,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 3)')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 256)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs for training (default: 50)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
parser.add_argument('--sample-seen', type=bool, default=False)

args = parser.parse_args()
assert args.output_dir is not None
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")

def load_checkpoint(model_kwargs, filename="outputs/outputnewmore/checkpoints/07-03-2019-21:47:26-200.m"):
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer

#samples from same classes it was trained on, as a sanity check
def sample_seen(model_kwargs, loaders, datasets):
    model, optimizer = load_checkpoint(model_kwargs)
    model.eval()
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders
    test_batch = next(iter(train_loader))
    with torch.no_grad():
        inputs = Variable(test_batch.cuda())
    samples = model.sample_conditioned(inputs)
    save_path = os.path.join(args.output_dir, 'figures/' + "test.png")
    save_test_grid(inputs, samples, save_path)
    
def run(model, optimizer, loaders, datasets):
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders
    test_batch = next(iter(test_loader))
    the_batch = next(iter(train_loader))
    mnist_test_batch = load_mnist_test_batch(args.mnist_data_dir, args.batch_size)

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # initial weighting for two term loss
    alpha = 1
    # main training loop
    tbar = tqdm(range(args.epochs))
    vbs = []
    for epoch in tbar:

        # train step (iterate once over training data)
        model.train()
        running_vlb = 0
        for batch in train_loader:
            #batch = deepcopy(r_batch)
            #random_bin = 0.5 #np.random.random()
            #batch[batch < random_bin] = 0
            #batch[batch >= random_bin] = 1
            inputs = Variable(batch.cuda())
            vlb = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb


        # update running lower bound
        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)
        logging.info(s)
        vbs.append(s)
        print(s)

        # reduce weight
        alpha *= 0.5

        #evaluate on test set by sampling conditioned on contexts
        model.eval()
        if (epoch + 1) % viz_interval == 0 or epoch == 0:
            #t_batch = deepcopy(test_batch)
            #random_bin = 0.5 #np.random.random()
            #t_batch[t_batch < random_bin] = 0
            #t_batch[t_batch >= random_bin] = 1
            with torch.no_grad():
                inputs = Variable(batch.cuda())
            samples = model.sample_conditioned(inputs)
            filename = time_stamp + '-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            save_test_grid(inputs, samples, save_path)
            #unseen Omniglot
            filename = time_stamp + '-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            with torch.no_grad():
                inputs = Variable(test_batch.cuda())
            samples = model.sample_conditioned(inputs)
            save_test_grid(inputs, samples, save_path)

            #unseen MNIST
            filename = time_stamp + '-mnist-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            with torch.no_grad():
                inputs = Variable(mnist_test_batch.cuda())
            samples = model.sample_conditioned(inputs)
            save_test_grid(inputs, samples, save_path)

        #checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            filename = time_stamp + '-{}.m'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'checkpoints/' + filename)
            model.save(optimizer, save_path)

    logging.info(vbs)

def main():
    # create datasets
    train_dataset = OmniglotSetsDataset(data_dir=args.data_dir, split='train',
                                        augment=True, sample_size=args.sample_size)
    test_dataset = OmniglotSetsDataset(data_dir=args.data_dir, split='test', sample_size=args.sample_size)
    datasets = (train_dataset, test_dataset)

    # create loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)

    n_features = 256 * 4 * 4  # output shape of convolutional encoder
    # create model
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.elu,
        'print_vars': args.print_vars
    }
    
    if args.sample_seen:
        sample_seen(model_kwargs, loaders, datasets)
    else:
        model = Statistician(**model_kwargs)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #model, optimizer = load_checkpoint(model_kwargs)
        run(model, optimizer, loaders, datasets)


if __name__ == '__main__':
    main()
