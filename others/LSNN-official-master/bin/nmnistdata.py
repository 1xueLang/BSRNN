import torch

from torch.utils.data import DataLoader
from spikingjelly.datasets import dvs128_gesture, n_mnist

def n_mnist_dataset(data_dir, T, batch_size, test_batch_size):
    train_set = DataLoader(
        dataset=n_mnist.NMNIST(
            os.path.join(data_dir, 'n-mnist'), train=True, data_type='frame', frames_number=T, split_by='number'
        ),
        batch_size=batch_size, shuffle=True
    )
    test_set = DataLoader(
        dataset=n_mnist.NMNIST(
            os.path.join(data_dir, 'n-mnist'), train=False, data_type='frame', frames_number=T, split_by='number'
        ),
        batch_size=test_batch_size
    )
    return train_set, test_set