import yaml
import time

from model.mobilevit import *
from preprocessing import train_valid_split
from transform import *

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets

def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((64, 64), dims, channels, num_classes=10, expansion=2)

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
    Read Configurations and Hyperparameters
    '''
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    print(config)

    ### Hyperparameters Configurations

    BATCH_SIZE = config['batch_size']
    N_EPOCH = config['epoch']
    LEARNING_RATE = config['learning_rate']

    ### Experiment Configurations
    dataset = config['dataset']
    training_set_allocation = config['train_allocation']
    model_size = config['model_size']
    train = config['training']
    unpruned_path = config['unpruned_save_path']
    test = config['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Data Loading
    '''
    if dataset == 'cifar10':
        print('Cifar10 Dataset Selected')
        cifar_data = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=standard_transform)
        cifar_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)

    # Split Training set into training and validation
    train_set, valid_set = train_valid_split(cifar_data, training_set_allocation)

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
    Model Instatiation - MobilViT
    '''
    if model_size == 'xs': model = mobilevit_xs()
    elif model_size == 'xxs': model = mobilevit_xxs()
    else:
        if model_size != 's':
            print('Model Size does not exist - Default selected')
            model_size = 's'
        model = mobilevit_s()

    model = model.to(device)    # Decide between GPU and CPU
    print(f'MobileViT {model_size}: {count_parameters(model)} parameters')

    '''
    Training
    '''
    if (train):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(),
                                lr = LEARNING_RATE)

        ### Define Lists for visualizations
        train_loss_plot = []
        train_acc_plot = []
        valid_loss_plot = []
        valid_acc_plot = []

        print("Begin Training")
        t1 = time.perf_counter()
        for epoch in range(N_EPOCH):
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
    
            train_loss_plot.append(round(train_loss / len(train_loader), 4))
            train_acc_plot.append(round(train_correct / train_total, 4))
            valid_loss_plot.append(round(valid_loss / len(valid_loader), 4))
            valid_acc_plot.append(round(valid_correct / valid_total, 4))
        
        t2 = time.perf_counter()

        print(f"Finished Training in {int(t2 - t1)} seconds")

        torch.save(model.state_dict(), unpruned_path)

    '''
    Testing
    '''
    if (test):
        print("Begin Testing")
        model.load_state_dict(torch.load(unpruned_path))
        
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for index, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {test_total} test images: {100 * test_correct // test_total} %')
