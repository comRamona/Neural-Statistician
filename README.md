## About

This repo contains our experiments and extensions based on the model outlined in [Towards a Neural Statistician (Edwards & Storkey, ICLR 2017)](https://arxiv.org/abs/1606.02185).
The base PyTorch implementation is provided by https://github.com/conormdurkan/neural-statistician Copyright (c) 2017-2018 Conor Durkan.


```
conda install pytorch
conda install torchvision -c pytorch 
```

# Neural Statistician

The code was written using Python 3.6, and currently works only with a GPU setup. Package requirements should be easy to handle using Conda and/or pip.

## Structure
The four main experiments are each located in their own subdirectory. Each subdirectory contains the following five files:

- ```data.py``` Methods for creating datasets, and custom providers leveraging PyTorch's ```Dataset``` class. 
- ```nets.py``` Subnetworks used in Statistician model, implemented as subclasses of ```torch.nn.Module```. These correspond to the shared encoder, statistic network, inference network, latent decoder, and observation decoder referred to in the paper.
- ```model.py``` Implementation of the Neural Statistician model.
- ```plot.py``` Utility functions for plotting results.
- ```run.py``` Driver program with tweakable command line args. 

The ```spatial```, ```omniglot```, and ```faces``` directories also contain an additional ```create.py``` file used for data preprocessing.

## Data
The following data sets are used:

- 28 x 28 greyscale Omniglot from [here](https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT)
- MNIST from [here](http://yann.lecun.com/exdb/mnist/)
- aligned YouTube Faces from [here](https://www.cs.tau.ac.il/~wolf/ytfaces/)

Further preprocessing is needed. This is detailed below.

## Usage 
The following commands require you to be in the corresponding experiment directories. Checkpoints and figures are saved at intervals to a specified output directory. To save disk space, you can opt to save these only on the final epoch of training.

#### Synthetic

- ```python synthrun.py --output-dir=output```

#### Spatial MNIST

- ```python spatialcreate.py --data-dir=mnist-data```

```mnist-data``` must contain gzipped MNIST files. The above command will pickle two files to ```mnist-data/spatial```.

- ```python spatialrun.py --output-dir=output --data-dir=/mnist-data/spatial```

#### Omniglot

- ```python omnicreate.py --data-dir=omniglot-data```

```omniglot-data``` must contain the file ```chardata.mat```. The above command will pickle a data file to the ```omniglot-data``` directory.

- ```python omnirun.py --output-dir=output --data-dir=omniglot-data --mnist-data-dir=mnist-data```

The MNIST data directory needs to be specified here for testing of the model on unseen data.

#### YouTube Faces

- ```python facescreate.py --data-dir=faces-data```

```faces-data``` must contain the original uncropped images, with an unaltered directory structure as given by the download. The above command will create another directory ```faces-data/64```, with the same structure, but where images are cropped to 64 x 64. Note that this operation will take significant disk space (about 1GB).

- ```python facesrun.py --output-dir=output --data-dir=faces/64```

## Results
#### Synthetic
<p align="center">
  <img width="460" height="460" src="https://github.com/comRamona/Neural-Statistician/blob/master/synthetic/output/figures/cluster2.png">
</p>
<p align="left">
  <img width="460" height="351" src="https://github.com/comRamona/Neural-Statistician/blob/master/synthetic/output/figures/mean2.png">
</p>
<p align="right">
  <img width="460" height="351" src="https://github.com/comRamona/Neural-Statistician/blob/master/synthetic/output/figures/variance2.png">
</p>

#### Spatial MNIST
##### Inputs
<p align="center">
  <img width="460" height="35.5" src="https://github.com/comRamona/Neural-Statistician/blob/master/spatial/input.png">
</p>

##### Outputs after 1,10,50 and 100 epochs
<p align="center">
  <img width="460" height="35.5" src="https://github.com/comRamona/Neural-Statistician/blob/master/spatial/ezgif.com-gif-maker.gif">
</p>

#### Omniglot
<p align="center">
  <img width="460" height="94.5" src="https://github.com/comRamona/Neural-Statistician/blob/master/omniglot/figures/oneshot.png">
</p>
<p align="center">
  <img width="460" height="94.5" src="https://github.com/comRamona/Neural-Statistician/blob/master/omniglot/figures/1shotomni.png">
</p>

#### YouTube Faces
<p align="center">
  <img width="460" height="460" src="https://github.com/comRamona/Neural-Statistician/blob/master/faces/ezgif.com-gif-maker.gif">
</p>

## Contact
Please use the [issues tracker](https://github.com/conormdurkan/neural-statistician/issues) for questions, problems etc.

## License

MIT License

Copyright (c) 2017-2018 Conor Durkan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
