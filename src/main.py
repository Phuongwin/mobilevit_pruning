from model.mobilevit import mobilevit_s
from transform import *

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

if __name__ == "__main__":

    '''
    Experiment Configurations
    '''
    BATCH_SIZE = 2
    EPOCH = 2
    LEARNING_RATE = 0.003
    MOMENTUM = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Data Loading
    '''
    cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=standard_transform)

    ###TODO: Split training into train/validation

    train_loader = torch.utils.data.DataLoader(cifar_train,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True
                                              )

    # valid_loader = torch.utils.data.DataLoader(valid_test,
    #                                            batch_size = BATCH_SIZE,
    #                                            num_workers = 2,
    #                                            drop_last = True,
    #                                            shuffle = True
    #                                           )

    cifar_test = datasets.CIFAR10('./data', train=False, download=True, transform=None)

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
    model = mobilevit_s()
    print(model)

    '''
    Training
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr = LEARNING_RATE,
                          momentum=MOMENTUM)

    for epoch in range(EPOCH):
        print(f"Epoch {epoch+1}")
        train_loss = 0.0
        train_correct = 0
        train_total = 0

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

            print(f'Epoch {epoch + 1} \
                    Training Loss: {(train_loss / len(train_loader)):.4f} \
                    Training Acc: {(train_correct / train_total):.4f}')
    '''
    Testing
    '''