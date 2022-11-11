[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

### News
[2022 11 09] This repo was born.

[2022 11 10] Team assembled.

### Introduction
This code base is for the hackathon project.


### How to download the data and prepare for it:
1. download the data from the link (https://drive.google.com/drive/folders/1kKKxA0F8Vcm3042TB-Qc_oF-RifNaYoT) and unzip everything
2. unzip the data folder and all of the npy files in each subfolder and move your data to the same folder where the code is, like this:

```
your/path/to/cloned/github/repo/
└───data
│     └──np
│        └──train
│        └──test
│───code
│───visulisation

```


### Install the following libraries:
argparse, pytorch1.4, numpy


### Run the code in your directory of mnist_robust/code/
```
python run.py 
--seed 1024 # random seed
--net 'moe' # calling mixture of model based UNet
--lr 0.001 # learning rate
--temp 2,0 # temperature scaling to soft the output prob
--width 8 # number of filters in the 1st conv/mlp layer
--loss_fun 'dice' # loss function between dice or ce
--train_noise 0 # 0 for no aug on train, 1 for gaussian noise on train
--test_noise 1 # 0 for no aug on test; 
                 1 for gaussian noise sigma 0.5; 
                 2 for gaussian noise sigma 1.0; 
                 3 for gaussian blurr with sigma 0.5;
                 4 for gaussian blurr with sigma 0.7;
                 5 for jigsaw images;
                 6 for new class image
--epsilon 5 # when larger than 0, it will use gradient attack,
              this is the strength of the FGSM attack
              
```

### Hyper-parameters:
Under constructions.
