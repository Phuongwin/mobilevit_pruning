# Pruning MobileViT Architecture

Purpose: Small scale experiment involving Apple's MobileVit Architecture with common pruning techniques typically used in Convolutional Neural Networks. The experiment focuses on analyzing whether the MobileViT architecture will benefit further from compression of weights. 

This repository satisfies the requirements of the final project for the course 525.733 Deep Learning for Computer Vision at Johns Hopkins University.

# Building
This repository uses pipenv as a virutal environment to run the source code. To create the environment, perform the following:

```
cd mobilevit_pruning
pipenv install
```

A Pipfile.lock file should appear which signifies the creation of the environment. To activate the environment, run the following:

```
pipenv shell
```

All dependencies should be install with Python 3.9.

Data
All datasets used in experimentation were downloaded from the torchvision libraries. If the datasets do not exist within the specified repository, then the script will automatically download the datasets.

# Usage
## Training and testing
Configuring experiments have been simplified through the usage of a config.yaml file. Opening the file introduces multiple configuration parameters and hyperparameters to determine how to run the main training and testing script. Below each parameter is defined with default values associated.

### Configuration
```
dataset (String: 'cifar10')
    - Detemine what dataset to use.
model_size: (String: 'xxs')
    - Choice between three model sizes ('s', 'xs', 'xxs').
training: (Boolean: True)
    - Determines whether to perform training script.
test: (Boolean: True)
    - Determines whether to perform testing script.
```
### Hyperparameters
```
train_allocation: (Float: 0.2)
    - Percentage of dataset to be allocated towards the training set, the rest towards validation.
epoch: (Integer: 2)
    - Defines number of passes through entire training set.
batch_size: (Integer: 128)
    - Defines number of training examples utilized in one iteration.
learning_rate: (Float: 0.001)
    - Defines step size at each iteration while moving towards a minimum of loss function.
```

## Pruning
The pruning script utilizes the same config.yaml file as the training and testing script, however it remains its own entity. The pruning script leveages only two configuration parameters.
```
dataset (String: 'cifar10')
model_size: (String: 'xxs')
```

Once configurations and hyperparameters are desired values, run either scripts from the root directory of the repository as such:
```
python ./src/main.py
or
python ./src/prune.py
```
**Note:** Running ```prune.py``` requires an unpruned weight file of the exact configuration inputted by the user. The main training and testing script will save files in the form:

```unpruned_{model_size}_{device}_weights_{dataset_name}.pth```

```prune.py``` will read ```model_size```, ```dataset_name``` from config.yml while automatically selecting ```device``` depending on what is available.