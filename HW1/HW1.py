import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

"""
Download and load the data
"""
def loadData():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3*32*32, 100)
        self.fc2 = torch.nn.Linear(100, 25)
        self.dropout = torch.nn.Dropout(0.1)
        self.output = torch.nn.Linear(25, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


def test(net, dataset, criterion, device):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in dataset:
            images, labels = data[0].view(-1, np.array(data[0].size()[1:]).prod()).to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, total_loss/len(dataset)


def train(net, train_dataset, criterion, optimizer, device):
    # for epoch in range(10):  # loop over the dataset multiple times
    # running_loss = 0.0
    for i, data in enumerate(train_dataset, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].view(-1, np.array(data[0].size()[1:]).prod()).to(device)
        labels = data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 1000 == 999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 1000))
        #     running_loss = 0.0


def main():
    train_loader, test_loader, classes = loadData()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("Using Device %s" % device)
    for epoch in range(1):
        train(net, train_loader, criterion, optimizer, device)
        train_accuracy, train_loss = test(net, train_loader, criterion, device)
        test_accuracy, test_loss = test(net, test_loader, criterion, device)
        print("### Epoch {}\ttrain accuracy: {}%\ttrain loss: {}\ttest accuracy: {}%\ttest loss:{}"
              .format(epoch+1, train_accuracy, train_loss, test_accuracy, test_loss))

    print("Finish!")


main()