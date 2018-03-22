#!/bin/bash

conda create -y -n torch python=3
source activate torch
conda install -y pytorch torchvision -c pytorch
git clone https://github.com/srm-soumya/cloud-check.git
mkdir data
python cifar10_classifier.py
source deactivate
