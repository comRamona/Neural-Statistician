from collections import Counter
from zipfile import ZipFile
from tqdm import tqdm
import random as rn
import os
import requests
import os
import sys
from urllib.request import urlretrieve
import numpy as np
from nltk import word_tokenize
#import pdb
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='bookNS2.log')
class GloveMatrix(object):
    """
    Downloads and loads GloVe matrix.
    """
    #https://nlp.stanford.edu/data/glove.840B.300d.zip
    def __init__(self):
        self.glove_url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        self.file_name = "/homes/rgc35/Desktop/neural-statistician/SentEval/glove.840B.300d.zip"
        self.dest = "/homes/rgc35/Desktop/neural-statistician/SentEval/glove.840B.300d"
        self.download_glove()
        embedding_index = self.load_matrix()
        self.EMBEDDING_DIM = 300
        print("Done")
        logging.debug("Done")
        
    def download_glove(self):
        if not os.path.exists("/homes/rgc35/Desktop/neural-statistician/SentEval/glove.840B.300d/glove.840B.300d.txt"):
            if os.path.exists(self.file_name):
                self.unzip_file(self.file_name, self.dest)
            else:
                urlretrieve(self.glove_url, self.file_name, self.reporthook)
                self.unzip_file(self.file_name, self.dest)
                
    def load_matrix(self):       
        print("Loading embedding matrix")
        logging.debug("Loading embedding matrix")
        self.embedding_index = {}
        with open('/homes/rgc35/Desktop/neural-statistician/SentEval/glove.840B.300d/glove.840B.300d.txt', "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                values = line.split()
                if len(values) > 1:
                    #pdb.set_trace()
                    try:
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        self.embedding_index[word] = coefs
                    except Exception as e:
                        pass

    def get_index(self):
        return self.embedding_index    

    def unzip_file(self, file_name, dest):
        print("Unzipping file...")
        zipTest = ZipFile(file_name)
        zipTest.extractall(dest)

    def download_file(self, url, file_name):
        print("Downloading file...")
        urlretriseve(url, file_name, reporthook)

    def reporthook(self, blocknum, blocksize, totalsize):
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

class TextEmbedder(object):
    """
    TextEmbedder returning word embeddings, using given GloVe matrix.
    """
    def __init__(self, glove_matrix):
        self.embedding_index = glove_matrix.embedding_index

    def get_any(self,word):
         return self.embedding_index.get(word, np.zeros(0)).astype(np.float32)
        
    def get_zero(self):
        return np.zeros(300).astype(np.float32)
    
    def get_sentence_embedding(self, sent, sent_length = 40):
        sent_vec = np.zeros((sent_length, 300))
        embs = [self.embedding_index.get(word, self.get_zero()) for word in sent[:sent_length]]
        sent_vec[:len(embs),:] = np.array(embs)
        return sent_vec