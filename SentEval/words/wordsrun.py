import argparse
import os
import time

from wordsdata import WikipediaDataset
from wordsmodel import Statistician
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size (of datasets) for training (default: 100)')

parser.add_argument('--n-stochastic', type=int, default=3,
                    help='number of z variables in hierarchy (default: 3)')
parser.add_argument('--z-dim', type=int, default=16,
                    help='dimension of z variables (default: 16)')
parser.add_argument('--c-dim', type=int, default=512,
                    help='dimension of c variables (default: 512)')


parser.add_argument('--n-hidden-statistic', type=int, default=3,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=512,
                    help='dimension of hidden layers in statistic network (default: 512)')
parser.add_argument('--n-hidden', type=int, default=3,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 3)')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 512)')



parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for training (default: 100)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
parser.add_argument('--show-plots', type=bool, default=False,
                    help='Display figues instead of saving '
                         '(default: False)')
args = parser.parse_args()
assert args.output_dir is not None
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")

def run(model, optimizer, loaders, datasets, show_plots=False):
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # initial weighting for loss terms is (1 + alpha)
    alpha = 1

    # main training loop
    tbar = tqdm(range(args.epochs))
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for batch in train_loader:
            inputs = Variable(batch.cuda())
            vlb = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb

        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

    model.save(optimizer, "finalmodel2.m")


def main():
    train_dataset = WikipediaDataset(data_dir=args.data_dir, split='train')
    test_dataset = WikipediaDataset(data_dir=args.data_dir, split='test')
    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)

    # hardcoded sample_size and n_features when making Spatial MNIST dataset
    sample_size = 15 # sentence length mode
    n_features = 300 # n-dimensional word embedding vectors
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.relu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run(model, optimizer, loaders, datasets, args.show_plots)


if __name__ == '__main__':
    main()
