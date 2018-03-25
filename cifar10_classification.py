import math
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

gpu = True if torch.cuda.is_available() else False

def timeit(method):
    '''@timeit decorator'''
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        print(f'{method.__name__}, {(end - start)} secs')
        return result
    return timed

@timeit
def download_data():
    '''Downloads the data and applies transformation'''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    return train, test

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@timeit
def run_model(train, num_epochs=2):
    # Create data loaders
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)

    # Define the network
    net = Net()
    if gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=9e-1)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 200 == 199:
                print(f'Epoch: {epoch+1}, MB: {i+1}, Loss: {running_loss / 200}')
                running_loss = 0.0

    print('Finished Training')
    return net

def compute_results(test, net):
    test_loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0

    for data in test_loader:
        imgs, labels = data
        imgs = Variable(imgs.cuda()) if gpu else Variable(imgs)
        outputs = net(imgs)
        _, preds = torch.max(outputs.data.cpu(), dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum()

    print(f'Overall Accuracy: {correct / total}\n\n')

    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    for data in test_loader:
        imgs, labels = data
        imgs = Variable(imgs.cuda()) if gpu else Variable(imgs)
        outputs = net(imgs)
        _, preds = torch.max(outputs.data.cpu(), dim=1)
        c = preds == labels
        for i in range(32):
            label = labels[i]
            class_total[label] += 1
            class_correct[label] += c[i]

    for i in range(10):
        print(f'Class: {classes[i]}, Score: {class_correct[i]/class_total[i]}')


def main():
    train, test = download_data()
    net = run_model(train, 2)
    compute_results(test, net)


if __name__ == '__main__':
    main()
