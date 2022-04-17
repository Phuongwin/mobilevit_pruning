from model.mobilevit import *
from preprocessing import train_valid_split
from transform import *

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((128, 128), dims, channels, num_classes=10, expansion=2)

def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((64, 64), dims, channels, num_classes=10)

def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((64, 64), dims, channels, num_classes=10)

if __name__ == "__main__":

    '''
    Experiment Configurations
    '''
    BATCH_SIZE = 128
    EPOCH = 2
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Data Loading
    '''
    cifar_data = datasets.CIFAR10('./data', train=True, download=True, transform=standard_transform)
    cifar_test = datasets.CIFAR10('./data', train=False, download=True, transform=None)

    # Split Training set into training and validation
    train_set, valid_set = train_valid_split(cifar_data, 0.8)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = BATCH_SIZE,
                                               num_workers = 2,
                                               drop_last = True,
                                               shuffle = True
                                              )

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size = BATCH_SIZE,
                                               num_workers = 2,
                                               drop_last = True,
                                               shuffle = True
                                              )

    test_loader = torch.utils.data.DataLoader(cifar_test,
                                              batch_size = BATCH_SIZE,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                             )
    '''
    Data Processing
    '''

    '''
    Model Instatiation - MobilViT
    '''
    model = mobilevit_xxs()
    print(model)

    '''
    Training
    '''
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr = LEARNING_RATE)

    for epoch in range(EPOCH):
        print(f"Epoch {epoch+1}")
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        model.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        model.eval()
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t Training Loss:   {(train_loss / len(train_loader)):.4f} \
                                     Training Acc:    {(train_correct / train_total):.4f}')
        print(f'Epoch {epoch + 1} \t Validation Loss: {(valid_loss / len(valid_loader)):.4f} \
                                     Validation Acc:  {(valid_correct / valid_total):.4f}')
    
    
    torch.save(model.state_dict(), "./saved_models/temp.pth")

    '''
    Testing
    '''
    model.load_state_dict(torch.load("./saved_models/temp.pth"))
    
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            images, labels = data

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {test_total} test images: {100 * test_correct // test_total} %')
