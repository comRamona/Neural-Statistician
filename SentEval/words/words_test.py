from wordsdata import WikipediaDataset
import os
import argparse
from torch.utils import data


parser = argparse.ArgumentParser(description='test')


parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')

args = parser.parse_args()
print(os.path.isdir(args.data_dir))

test_dataset = WikipediaDataset(data_dir=args.data_dir, split='test')

test_loader = data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=0, drop_last=True)

#a = next(iter(test_loader))
for i in range(20):
	a = next(iter(test_loader))
	print(a[0][0][0])
