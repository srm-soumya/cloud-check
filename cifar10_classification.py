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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@timeit
def run_model(train, num_epochs=2):
    # Create data loaders
    train_loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=2)

    # Define the network
    net = Net()
    if gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
            if i % 1000 == 999:
                print(f'Epoch: {epoch+1}, MB: {i+1}, Loss: {running_loss / 1000}')
                running_loss = 0.0

    print('Finished Training')
    return net

def compute_results(test, net):
    test_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0

    for data in test_loader:
        imgs, labels = data
        outputs = net(Variable(imgs))
        _, preds = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum()

    print(f'Overall Accuracy: {correct / total}\n\n')

    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    for data in test_loader:
        imgs, labels = data
        outputs = net(Variable(imgs))
        _, preds = torch.max(outputs.data, dim=1)
        c = preds == labels
        for i in range(4):
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
