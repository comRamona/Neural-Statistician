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


		batch = self.content[self.content_idx]

		for i in range(40):
			if batch[i][0]==0:
				break

		batch = batch[:i]
		self.content_idx += 1

		if self.content_idx == 10000:
			#print('done with file')
			self.content = None
			self.file_index += 1
			self.content_idx = 0

		batch = np.array(batch).astype(np.float32)
		return batch, len(batch)

	def __len__(self):
		return 1000000

# class SpatialMNISTDataset(data.Dataset):
#     def __init__(self, data_dir, split='train'):
#         splits = {
#             'train': slice(0, 60000),
#             'test': slice(60000, 70000)
#         }

#         spatial_path = os.path.join(data_dir, 'spatial.pkl')
#         with open(spatial_path, 'rb') as file:
#             spatial = pickle.load(file)

#         labels_path = os.path.join(data_dir, 'labels.pkl')
#         with open(labels_path, 'rb') as file:
#             labels = pickle.load(file)

#         self._spatial = np.array(spatial[splits[split]]).astype(np.float32)[:10000]
#         self._labels = np.array(labels[splits[split]])[:10000]

#         ix = self._labels[:, 1] != 1
#         self._spatial = self._spatial[ix]
#         self._labels = self._labels[ix]

#         assert len(self._spatial) == len(self._labels)
#         self._n = len(self._spatial)

#     def __getitem__(self, item):
#         return self._spatial[item], self._labels[item]

#     def __len__(self):
#         return self._n

