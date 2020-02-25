import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
from time import time
import matplotlib.pyplot as plt


def load_data(batch_size=128):
    """
    Download and load the data
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data.cifar10', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1)
        self.conv1_bn = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3)
        self.conv2_bn = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.conv3_bn = torch.nn.BatchNorm2d(128)
        
        self.fc1 = torch.nn.Linear(128 * 2 * 2, 120)
        self.fc1_bn = torch.nn.BatchNorm1d(120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc2_bn = torch.nn.BatchNorm1d(84)
        self.fc3 = torch.nn.Linear(84, 10)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout1d = torch.nn.Dropout(0.3)
        self.dropout2d = torch.nn.Dropout2d(0.3)
        
        self.conv1_output = None


    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        self.conv1_output = x
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.dropout2d(x)
        x = x.view(-1, self.flatten_feature(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout1d(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout1d(x)
        x = self.fc3(x)
        
        return x
    
    def flatten_feature(self, x):
        num_feature = 1
        for d in x.size()[1:]:
            num_feature *= d
        return num_feature


def test(net, inputs, device, criterion=None, flag='test'):
    net.eval()
    if flag == 'inference':
        with torch.no_grad():
            image = inputs.to(device)
            image = image.unsqueeze(0)
            output = net(image)
            _, predicted = torch.max(output.data, 1)
            return predicted.int(), net.conv1_output.cpu()
    else:
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in inputs:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_loss += criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total, float(total_loss/len(inputs))
    

def train(device, total_epoch):
#     writer = SummaryWriter()
    train_loader, test_loader, classes = load_data()
    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.to(device)
    best_test_accuracy = 0
    old_train_lost = 0
    start = time()
    for epoch in range(1, total_epoch+1):
        for i, data in enumerate(train_loader, 0):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_accuracy, train_loss = test(net, train_loader, device, criterion)
        test_accuracy, test_loss = test(net, test_loader, device, criterion)
        if epoch % 10 == 0:
            print("### Epoch [{}/{}]\t\ttrain accuracy: {}%\t\ttrain loss: {}\t\ttest accuracy: {}%\t\ttest loss:{}"
                  .format(epoch,total_epoch, train_accuracy, round(train_loss, 7), test_accuracy, round(test_loss, 4)))
#         writer.add_scalar('Test/Loss', test_loss, epoch)
#         writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
#         writer.add_scalar('Train/Loss', train_loss, epoch)
#         writer.add_scalar('Train/Accuracy', train_accuracy, epoch)


        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            torch.save(net.state_dict(), "./model/cnn.pt")
            # print("### Epoch {}\t\tSave Model fot test accuracy {}% ".format(epoch+1, best_test_accuracy))
        if abs(old_train_lost-train_loss) < 1e-5:
            print("Training almost converge in Epoch {}, STOP!".format(epoch))
    print("Finish in {}!".format(time()-start))
    print("The best Model for testing accuracy of {}% is saving in ./model".format(best_test_accuracy))
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convolutional Neural Network')
    parser.add_argument('mode', type=str, help='Mode: train / test')
    parser.add_argument('image', type=str, default='', nargs='?', help='Image file in test')
    parser.add_argument('--epoch', type=int, default=150, help='Total Number of Epoch')
    arg = parser.parse_args()

    # print(arg.mode, arg.image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using Device %s" % device)
    if arg.mode == 'train':
        train(device, arg.epoch)
    elif arg.mode == 'test':
        net = Net()
        net.to(device)
        net.load_state_dict(torch.load('./model/cnn.pt', map_location=device))
        print("Load network from ./model/cnn.pt")
        # Open image
        img = Image.open(arg.image)
        # convert image to Tensor
        transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_tensor = transform(img).float()
        label, conv1_output = test(net, image_tensor, device, flag='inference')
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        conv1_output = conv1_output / 2 + 0.5
        for id, output in enumerate(conv1_output):
            fig, axs = plt.subplots(8, 4, figsize=(1, 1), sharex=True,sharey=True)
            fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99, hspace = 0.1, wspace = 0.1)
            for i, ax in enumerate(axs.flatten()):
                ax.axis('off')
                ax.imshow(output[i], interpolation='hamming', cmap='gray')
            fig.savefig('CONV_rslt.png', dpi=500)
            
        print("Predict: %s" % classes[label])