import numpy as np
import os
import pickle

from torch.utils import data


class WikipediaDataset(data.Dataset):
	def __init__(self, data_dir, split="train"):

		self.path = data_dir
		#self.path = '/Users/tyler/Desktop/lent/advanced_ml/from_git_mar_5/Neural-Statistician/SentEval/words/from_cluster'
		self.file_index = 1
		self.content_idx = 0
		self.content = None
		pass

	def __getitem__(self, item):
		
		if self.content is None:
			this_file = self.path + '/%06i.pkl' % self.file_index
			#print(this_file)
			in_file = open(this_file, 'rb')
			self.content = pickle.load(in_file)

		batch = np.array(self.content[self.content_idx]).astype(np.float32)
		self.content_idx += 1

		if self.content_idx == 10000:
			#print('done with file')
			self.content = None
			self.file_index += 1
			self.content_idx = 0
		if self.file_index == 500:
			self.file_index = 1

		return batch

	def __len__(self):
		return 2000000 #2 million total sentence embeddings
