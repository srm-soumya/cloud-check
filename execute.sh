#!/bin/bash

conda create -y -n torch python=3
source activate torch
conda install -y pytorch torchvision -c pytorch
git clone https://github.com/srm-soumya/cloud-check.git && cd cloud-check
mkdir data
python cifar10_classification.py
source deactivate
