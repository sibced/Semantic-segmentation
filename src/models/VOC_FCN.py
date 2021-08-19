from ..data.voc_dataset import VOCDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.BatchNorm2d(out_channels), nn.ReLU())
def deconv(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding), nn.BatchNorm2d(out_channels), nn.ReLU())

class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        

        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True)  #Downsamples input image by size of 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True, ceil_mode=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)    #Upsamples input image by size of 2x2

        #Convolutional network
        self.conv1 = conv(3, 21, kernel_size=4, stride=2)   #Downsamples input image by size of 2x2
        self.conv2 = conv(21, 21, kernel_size=4, stride=2)
        self.conv3 = conv(21, 21, kernel_size=4, stride=2)
        self.conv4 = conv(21, 21, kernel_size=4, stride=2)

        #Deconvolutional network
        self.deconv4 = deconv(21, 21, kernel_size=4, stride=2)  #Upsamples input image by size of 2x2
        self.deconv3 = deconv(21, 21, kernel_size=4, stride=2)
        self.deconv2 = deconv(21, 21, kernel_size=4, stride=2)
        self.deconv1 = deconv(21, 21, kernel_size=4, stride=2)
                                 

    def forward(self, x):

        x = self.conv1(x)
        size1 = x.size()
        x, indices1 = self.pool1(x)

        x = self.conv2(x)
        size2 = x.size()
        x, indices2 = self.pool2(x)
        
        x = self.conv3(x)
        size3 = x.size()
        x, indices3 = self.pool1(x)

        x = self.conv4(x)
        size4 = x.size()
        x, indices4 = self.pool1(x)

        x = self.unpool(x, indices=indices4, output_size=size4)
        x = self.deconv4(x)

        x = self.unpool(x, indices=indices3, output_size=size3)
        x = self.deconv3(x)

        x = self.unpool(x, indices=indices2, output_size=size2)
        x = self.deconv2(x)

        x = self.unpool(x, indices=indices1, output_size=size1)
        x = self.deconv1(x)

        return x

DIR = r"C:\Users\cedri\Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
train_set = VOCDataset(DIR)
test_set = VOCDataset(DIR, train=False)


train_ldr = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
test_ldr = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


device = 'cuda:0'
model = net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(num_epochs):

    train_loss = torch.zeros((num_epochs))
    test_loss = torch.zeros((num_epochs))
    test_acc = torch.zeros((num_epochs))

    for i in range(num_epochs):
        print(f'epoch ({i+1:2}/{num_epochs:2})', end='\r')
        model.train()
        print("Training")
        for m, l in enumerate(train_ldr):
            print("Image " + str(m+1))
            for k, (x, y) in enumerate(l):
                #print(f'epoch ({i+1:2}/{num_epochs:2}) - batch ({k+1:3}/{len(train_ldr):3})', end='\r')
                outputs = model(x)
                loss = criterion(outputs, y)

                train_loss[i] += loss.detach().sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('')
        print(f'epoch {i+1:2}/{num_epochs:2} - evaluation')
        with torch.no_grad():
            print("Testing")
            model.eval()
            accuracy = 0.0
            for x, y in test_ldr:
                inputs = x
                targets = y

                outputs = model(inputs)

                loss = criterion(outputs, targets)
                test_loss[i] += loss.sum()

                # per - pixel accuracy
                predicted = outputs.argmax(1)
                accuracy += (predicted == targets).float().mean()
            test_acc[i] = accuracy / len(test_ldr)

    train_loss /= (num_epochs * train_ldr.batch_size * len(train_ldr))
    test_loss /= (num_epochs * test_ldr.batch_size * len(test_ldr))

    return train_loss, test_loss, test_acc

if __name__ == '__main__':
    num_epochs = 100
    print(f'training for {num_epochs} epochs')
    epochs_train_loss, epochs_test_loss, test_acc = train(num_epochs)
    torch.save(model.state_dict(), "./PascalVOClr1e-3.pth")

    print(f'{epochs_train_loss}\n{epochs_test_loss}\n{test_acc}')

    idx = range(0, num_epochs)
    plt.figure()
    plt.plot(idx, epochs_train_loss, label='training loss')
    plt.plot(idx, epochs_test_loss, label='testing loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(idx, test_acc, 'k', label='testing accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()