### Introduction
This repository is an implementation for a submission to kannada mnist classification
using a semi-supervised learning with 
best validation acc on Dig-MNIST at 86% and 
best testing acc at 98.72% (private) and 98.74%(public).

We tested with a simple 3 layers CNN and a VGG-like network and the training strategy is:
1) supervised learning on labelled data
2) unsupervised learning on unlabelled data after heavy data augmentation and high confident pseudo labels.
3) when use data augmentation (zoom, rotation, affine) on labelled data, it seems like small lr 0.001 works better, 
without data augmentation on labelled data, seems like large lr 0.01 works well.
4) polynominal learning rate decay

### Installation and Usage
This repository is based on PyTorch 1.4 and it needs numpy, os, maths, random, pandas libraries.
To train the model with default hyper parameters:

   ```shell
   cd kannada_exp
   python run.py --path 'path/to/your/kannada_dataset' 
   ```

However, those hyper parameters might not be the most optimal ones.

### Some attempts I tried but somehow didn't improve results:
1. large model including vgg and resent18, resnet80, however, it didn't improve the validation accuracy and they are very slow to train so not used due to time constraints.

2. Complicated data augmentation could even bring the performance down 
if not used carefully with this data set.

3. Ensemble (Stochastic weight average, different initialisation, different dropouts) improved validation acc on Dig-MNIST to 84%, however it doesn't help
with the test somehow. I tried ensemble on different model architectures and different
model capacities.

4. I also tried other regularisation such as consistency on different views of input, et al. might need more hp search


### Some issues I encountered:
1. I was using Dig-MNIST for validation so I didn't realise until very late 
that the real testing acc is much higher than the validation acc. 
This might imply the labels in Dig-MNIST might be wrong (e.g. seems like those are
acquired from non-native speakers)

