from __future__ import absolute_import, division, unicode_literals
from embeddings2 import GloveMatrix, TextEmbedder
from wordsmodel import Statistician
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from utils import kl_diagnormal_diagnormal, pytorch_wass, cosine_sim
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


def kl_similarity(s1, s2, sim_metric = cosine_sim):
    n = int(s1.shape[0] / 2)
    c_mean1, c_logvar1 = s1[0:n], s1[n:]
    c_mean2, c_logvar2 = s2[0:n], s2[n:]
    sim = sim_metric(c_mean1, c_logvar1, c_mean2,c_logvar2)
    if sim_metric == kl_diagnormal_diagnormal return 1 - sim else return sim


def load_checkpoint(filename="mm.m"):
    sample_size = 40
    n_features = 300
    model_kwargs = {
        'batch_size': 100,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': 64,
        'n_hidden_statistic': 3,
        'hidden_dim_statistic': 256,
        'n_stochastic': 3,
        'z_dim': 2,
        'n_hidden': 3,
        'hidden_dim': 256,
        'nonlinearity': F.relu,
        'print_vars': False
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), 1e-3)
    model_dict = model.state_dict()
    checkpoint = torch.load(filename)
    pretrained_dict  = checkpoint['model_state'] 
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    #model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer


class fake_model():
    def __init__(self):
        pass
    def sent_embedding(self, vect):
        c1 = np.mean(np.random.random((50, 300)), 0)[:50]
        v1 = np.var(np.log(np.random.random((50,300))),0)[:50]
        return np.concatenate([c1,v1])
    
#model = fake_model()    
model, optimizer = load_checkpoint()

gm = GloveMatrix()
te = TextEmbedder(gm)

# SentEval prepare and batcher
def prepare(params, samples):
    params.wvec_dim = 300
    params.similarity = kl_similarity
    return

def batcher(params, batch, sent_length = 40):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    model.eval()
    for sent in batch:
        sentvect = te.get_sentence_embedding(sent, sent_length)
        with torch.no_grad():
            inputs = Variable(torch.from_numpy(sentvect).float().cuda())
            mv = model.sent_embedding(inputs)
        embeddings.append(mv)

    embeddings = torch.stack(embeddings)
    #print(embeddings.shape)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 1, 'similarity' : kl_similarity}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = se.eval(transfer_tasks)
    print(results)