import torch

def train_valid_split(dataset, train_percentage: float = 0.8):
    
    dataset_size = dataset.__len__()

    train_count = int(dataset_size  * train_percentage)
    valid_count = dataset_size - train_count

    train_set, valid_set = torch.utils.data.random_split(dataset, [train_count, valid_count])

    print(f'Train Set: {train_set.__len__()}')
    print(f'Validation Set: {valid_set.__len__()}')
    print(f"Total: {train_set.__len__() + valid_set.__len__()}")

    return train_set, valid_set