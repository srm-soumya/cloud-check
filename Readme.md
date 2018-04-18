## What is Deep Learning?

Deep Learning(DL) is a branch of Machine Learning which tries to mimic the architecture of neurons in human brain and
calls them Artifical Neural Networks(ANN). Although ANNs have been there since the 1950s, the recent advancement with GPU based Infrastructure and availability of large datasets is what lead to the boom of DL Applications. We can see their implementation in day-to-day application like Facebook Auto Image Tags, Siri, Google Translate, Self-driving cars etc. DL has brought us a step closer to the ambition of achieving near human intelligence for machines.


## Deep Learning at Gramener

**Gramener** is a *Data Science* company. We started building DL skills 8 months back,
as a part of it's AI Labs Initiative.

We started with a team of 3 people, experimenting with different DL techniques and trying to build
models that could solve real-world problems. We have worked on a variety of problems like Image Classification, Object Detection in Videos, Facial Recognition etc. During this process, we got a chance to work with the different size of datasets ranging from 1-2 GB to 200-300 GB. Since the team is growing and the intent is to make DL as a practice in the organization, we thought of deciding on the infrastructure that will help us in:
- Training more people to work on DL Initiatives/Projects
- Experimenting with different kinds of DL problems and build prototype models for use
- Building pre-trained consumable models for client business needs
- Providing API's or building Web Applications on top of the model


## Setup

Keeping the above points in mind, we thought of evaluating these 4 Cloud Providers:
**Microsoft Azure**, **Amazon Web Services (AWS)**, **Google Compute Engine (GCE)** & **Paperspace**.

Although there are a few other vendors in the market like *FloydHub*, *ClusterOne* & *Crestle*, their DL workflow
does not suit our model of development.

*Problem*: Build a CNN model to classify Images for the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Choices

- Dataset: **CIFAR10** => Small enough to be treated as an experimental dataset.
- Framework: **PyTorch** => Is more pythonic compared to other DL frameworks, is available as a beta release.
- CNN Model: **VGG16** => Intuitive to understand

Trained the VGG16 Model for *5 Epochs* (Random choice), which gives an accuracy in the range of *55%-65%* which
is decent compared to the base accuracy of 10%.

## Experiment

Steps:
1. Download the executable script from the github repository: [execute.sh](https://github.com/srm-soumya/cloud-check/blob/master/execute.sh) to the cloud infrastructure.
2. Make it executable (chmod +x execute.sh)
3. Run the script (./execute.sh)

It downloads the CIFAR10 dataset, runs the VGG16 model for 5 Epochs & prints the result in the console.

## Results

We will evaluate different cloud providers based on:
- Ease of setting up the infrastructure
- Speed
- Cost

| Cloud Provider | Region     | GPU     | Image                         | Run time (secs) | Cost ($/hr) | Cost / run |
|----------------|------------|---------|-------------------------------|----------------:|-------------|------------|
| Paperspace     | NY2        | 1 P4000 | fast-ai AMI                   |             354 |       $0.40 |      $0.04 |
| Azure          | US-East    | 1 P100  | DSVM Ubuntu: NC6 Standard SDD |             176 |       $2.11 |      $0.10 |
| AWS            | US-East    | 1 V100  | ami-005a6c65, p3.2xlarge, HDD |             262 |       $3.06 |      $0.22 |
| GCE            | US-West-1b | 1 K80   | Ubuntu 16.04 with SSD 30 GB   |            1461 |       $0.58 |      $0.24 |
| Azure          | US-East    | 1 K80   | DSVM Ubuntu: NC6 Standard HDD |            1428 |       $0.92 |      $0.36 |
| AWS            | US-East    | 1 K80   | ami-005a6c65, p2.xlarge, HDD  |            1431 |       $0.90 |      $0.36 |

#### Paperspace
From the experiment results, Paperspace looks like the most viable option, with Cost/run 2.5x times compared to the
second best. However, it took me 4-5 attempts to ssh into the VM and also every time you create a new VM for
experimentation you have to pay an extra cost of at least $5 for storage.

#### Microsoft Azure
Azure comes with Data Science Virtual Machines (DSVMs) which is a pre-configured Virtual Machine for ML and DL workflows.
It is very easy to set up and start working on these machines. Among the major cloud providers, Azure NC6v2 DSVM with P100 GPUs
seems to be the best alternative. However, most of our time is spent in experimenting with the different models rather than training
production ready deep learning models, so we are looking for options where Cost/hour is the cheapest.

#### Google Compute Engine
GCE doesn't come with any pre-configured VMs for DL workflow, so we had to start a vanilla Ubuntu 16.04 server and installed
the required binaries and libraries for the experiment. Although the Cost/run is comparatively high for GCE compared to Azure & AWS,
it seems to be the cheapest solution available out there, which is good for team members to experiment and learn.

Based on the above considerations, we have decided to use GCE for experimentation and Azure NC_v2 series machines for
training larger models.

## Further Consideration:
Google has now come up with [Google Colaboratory](https://colab.research.google.com/), which allows anyone to start experimenting with DL models, giving free 12 hours of usage of Tesla K80 GPUs at a stretch. It is one of the best possible options available for people coming into DL field and who don't want to invest much on the infrastructure initially.

Azure, GCE, AWS are also adding latest Nvidia GPUs rapidly, will keep an eye on them and update the results based on that.

We at Gramener are also planning to purchase a few Gaming laptops and start building on them, will evaluate and publish the results as and when I get my hands dirty on them.
