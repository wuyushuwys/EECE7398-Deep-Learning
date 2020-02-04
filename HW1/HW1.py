import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np


train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)
train_loader = Data.DataLoader(dataset=train_data,batch_size=10, shuffle=False)

test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=False )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = torch.nn.Linear(3*32*32, 100)
        # # self.dropout1 = torch.nn.Dropout(0.5)
        # self.fc2 = torch.nn.Linear(100, 25)
        # # self.dropout2 = torch.nn.Dropout(0.25)
        # self.output = torch.nn.Linear(25, 10)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # # x = torch.flatten(self.dropout1(x), 1)
        # x = F.relu(self.fc2(x))
        # # x = self.dropout2(x)
        # x = self.output(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(dataset, loss_func):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loss = 0
        for data in dataset:
            images, labels = data
            # images = images.view(-1, 32*32*3)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += loss_func(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(' Accuracy: {}% , Test Loss: {}'.format(correct/total*100, loss/len(dataset)))


net = Net()
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for step, data in enumerate(train_loader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs = inputs.view(-1, np.array(inputs.size()[1:]).prod())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        if step % 2000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d]' % (epoch + 1, step + 1))

            # print('[%d, %5d] Training loss: %.3f' % (epoch + 1, step + 1, loss.item()))
            test(train_loader, loss_func)
            running_loss = 0.0


print('Finished Training')