from __future__ import absolute_import, division, unicode_literals
from embeddings import GloveMatrix, TextEmbedder
from wordsmodel import Statistician
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from utils import kl_diagnormal_diagnormal
import sys
import io
import numpy as np
import logging

PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_VEC = 'glove.6B'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

gm = GloveMatrix()
te = TextEmbedder(gm)

def bow_prepare(params, samples):
    pass

def bow_batcher(params, batch, sent_length = 50):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    model.eval()
    for sent in batch:
        sentvect = te.get_sentence_embedding(sent, sent_length)
        emb = np.mean(sentvect, 0)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    return embeddings



# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, bow_batcher, bow_prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']#,

    results = se.eval(transfer_tasks)
    print(results)