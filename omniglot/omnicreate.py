import argparse
import numpy as np
import os
import pickle
import sys
from urllib.request import urlretrieve

from downloading import download_file

from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, type=str, default=None)
parser.add_argument('--mnist-data-dir', required=True, type=str, default=None)
args = parser.parse_args()
assert (args.data_dir is not None) 
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
if not os.path.exists(args.mnist_data_dir):
    os.makedirs(args.mnist_data_dir)
omniglot_source = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/"
omniglot_file_name = "chardata.mat"
mnist_file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 
                    't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
mnist_source = "http://yann.lecun.com/exdb/mnist/"

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))
                   
def load():
    # download if needed
    download_file(omniglot_source, args.data_dir, "chardata.mat")                     
    # load data
    file = os.path.join(args.data_dir, 'chardata.mat')
    data = loadmat(file)

    # data is in train/test split so read separately
    train_images = data['data'].astype(np.float32).T
    train_alphabets = np.argmax(data['target'].astype(np.float32).T, axis=1)
    train_characters = data['targetchar'].astype(np.float32)

    test_images = data['testdata'].astype(np.float32).T
    test_alphabets = np.argmax(data['testtarget'].astype(np.float32).T, axis=1)
    test_characters = data['testtargetchar'].astype(np.float32)

    # combine train and test data
    images = np.concatenate([train_images, test_images], axis=0)
    alphabets = np.concatenate([train_alphabets, test_alphabets], axis=0)
    characters = np.concatenate([np.ravel(train_characters),
                                 np.ravel(test_characters)], axis=0)
    data = (images, alphabets, characters)

    return data


def modify(data):
    # We don't care about alphabets, so combine all alphabets
    # into a single character ID.
    # First collect all unique (alphabet, character) pairs.
    images, alphabets, characters = data
    unique_alphabet_character_pairs = list(set(zip(alphabets, characters)))

    # Now assign each pair an ID
    ids = np.asarray([unique_alphabet_character_pairs.index((alphabet, character))
                      for (alphabet, character) in zip(alphabets, characters)])

    # Now split into train(1200)/val(323)/test(100) by character
    train_images = images[ids < 1200]
    train_labels = ids[ids < 1200]
    val_images = images[(1200 <= ids) * (ids < 1523)]
    val_labels = ids[(1200 <= ids) * (ids < 1523)]
    test_images = images[1523 <= ids]
    test_labels = ids[1523 <= ids]

    split_data = (train_images, train_labels, val_images,
                  val_labels, test_images, test_labels)

    return split_data


def save(data):
    savepath = os.path.join(args.data_dir, 'train_val_test_split.pkl')
    with open(savepath, 'wb') as file:
        pickle.dump(data, file)


def main():
    #download omniglot
    data = load()
    modified_data = modify(data)
    save(modified_data)
    #download mnist
    for f in mnist_file_names:
        download_file(mnist_source, args.mnist_data_dir, f)

if __name__ == '__main__':
    main()