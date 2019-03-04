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

class GloveMatrix(object):
    """
    Downloads and loads GloVe matrix.
    """
    def __init__(self):
        self.glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        self.file_name = "glove.6B.zip"
        self.dest = "glove.6B"
        self.download_glove()
        embedding_index = self.load_matrix()
        self.EMBEDDING_DIM = 300
        print("Done")
        
    def download_glove(self):
        if not os.path.exists("glove.6B/glove.6B.300d.txt"):
            if os.path.exists(self.file_name):
                self.unzip_file(self.file_name, self.dest)
            else:
                urlretrieve(self.glove_url, self.file_name, self.reporthook)
                self.unzip_file(self.file_name, self.dest)
                
    def load_matrix(self):       
        print("Loading embedding matrix")
        self.embedding_index = {}
        with open('glove.6B/glove.6B.300d.txt', "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                values = line.split()
                if len(values) > 0:
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.embedding_index[word] = coefs

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
    
    def get_sentence_embedding(self, sent, sent_length = 50):
        sent_vec = []
        for word in sent[:sent_length]:
            emb = self.get_any(word)
            if emb.shape[0] != 0:
                sent_vec.append(emb)
        n = len(sent_vec)
        for w in range(n, sent_length):
            sent_vec.append(self.get_zero())
        return np.vstack(sent_vec)