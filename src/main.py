# Generic Python Libraries
import yaml
import time

# Necessary import files
from model.mobilevit import *
from preprocessing import *
from transform import *
from visualize import *

# Machine Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

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
    dataset_name = config['dataset'].lower()
    model_size = config['model_size'].lower()
    training_set_allocation = config['train_allocation']
    train = config['training']
    test = config['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    unpruned_path = f"./saved_models/unpruned_{model_size}_weights_{device}_{dataset_name}.pth"

    '''
    Data Loading
    '''
    if dataset_name == 'cifar10':
        print('Cifar10 Dataset Selected')
        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=standard_transform)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)
    elif dataset_name == 'cifar100':
        print('Cifar100 Dataset Selected')
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=standard_transform)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=test_transform)

    # Split Training set into training and validation
    train_set, valid_set = train_valid_split(dataset_train, training_set_allocation)
    print(f'Test Set: {dataset_test.__len__()}')

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

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size = BATCH_SIZE,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                             )
    '''
    Model Instatiation - MobileViT
    '''
    if model_size == 'xs': 
        model = mobilevit_xs()
    elif model_size == 'xxs': 
        model = mobilevit_xxs()
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
            train_acc_plot.append(round(100 * (train_correct / train_total), 4))
            valid_loss_plot.append(round(valid_loss / len(valid_loader), 4))
            valid_acc_plot.append(round(100 * (valid_correct / valid_total), 4))
        
        t2 = time.perf_counter()

        print(f"Finished Training in {int(t2 - t1)} seconds")

        torch.save(model.state_dict(), unpruned_path)

        epoch_count = range(1, N_EPOCH + 1)
        plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, model_size, dataset_name ,'Accuracy')
        plot_train_validation(epoch_count, train_loss_plot, valid_loss_plot, model_size, dataset_name, 'Loss')

    '''
    Testing
    '''
    if (test):
        print("Begin Testing")
        model.load_state_dict(torch.load(unpruned_path))
        
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        print(f'Accuracy of unpruned network on {test_total} test images: {100 * test_correct // test_total}%')
