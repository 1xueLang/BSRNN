import scipy.io as scio
import numpy as np
import torchvision, os, torch
from torchvision import transforms

def mnist_dataset(data_dir, batch_size, test_batch_size, encoder):
    transform_train = transforms.Compose([
        # transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        # transforms.Normalize(0.1307, 0.3081),
        torch.flatten,
        encoder
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.1307, 0.3081),
        torch.flatten,
        encoder
    ])
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            transform=transform_train,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            transform=transform_test,
            download=True),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_data_loader, test_data_loader

def Fmnist_dataset(data_dir, batch_size, test_batch_size, encoder):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.2860, 0.3530),
            torch.flatten,
            encoder
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.2860, 0.3530),
            torch.flatten,
            encoder
        ])
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            transform=transform_train,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=False,
            transform=transform_test,
            download=True),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True)
    return train_data_loader, test_data_loader


class SpeechDigit(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.features = None
        self.labels = None
        self.load_data_mat()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        x = self.features[item].T
        y = self.labels[item] - 1
        x = self.transform(x) if self.transform else x
        y = self.target_transform(y) if self.target_transform else y
        return x, y
    
    def load_data_mat(self):
        data_mat = scio.loadmat(os.path.join(self.root, 'Speech100data.mat'))
        if self.train:
            self.features = np.array(data_mat['trainData'])
            self.labels = np.array(data_mat['train_labels'])
        else:
            self.features = np.array(data_mat['testData'])
            self.labels = np.array(data_mat['test_labels'])

def sdigit_dataset(data_dir, batch_size, test_batch_size):
    tr_set = SpeechDigit(data_dir)
    ts_set = SpeechDigit(data_dir, False)
    tr_loader = torch.utils.data.DataLoader(
        dataset=tr_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    ts_loader = torch.utils.data.DataLoader(
        dataset=ts_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2
    )
    return tr_loader, ts_loader

class ECG(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.features = None
        self.labels = None
        self.load_from_mat()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        x, y = self.features[item], self.labels[item]
        x = self.transform(x) if self.transform else x
        y = self.target_transform(y) if self.target_transform else y
        return x, y
    
    def load_from_mat(self):
        file_name = 'QTDB_' + ('train' if self.train else 'test') + '.mat'
        _, self.features, self.labels = self.convert_dataset_wtime(
            scio.loadmat(os.path.join(self.root, file_name))
        )
    
    @staticmethod
    def convert_dataset_wtime(mat_data):
        X = mat_data["x"]
        Y = mat_data["y"]
        t = mat_data["t"]
        Y = np.argmax(Y[:, :, :], axis=-1)
        d1,d2 =  t.shape
        # dt = np.zeros((size(t[:, 1]), size(t[1, :])))
        dt = np.zeros((d1,d2))
        for trace in range(d1):
            dt[trace, 0] = 1
            dt[trace, 1:] = t[trace, 1:] - t[trace, :-1]

        return dt, X, Y
    
def ecg_dataset(data_dir, batch_size, test_batch_size):
    tr_set = ECG(data_dir, train=True)
    ts_set = ECG(data_dir, train=False)
    tr_loader = torch.utils.data.DataLoader(
        dataset=tr_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    ts_loader = torch.utils.data.DataLoader(
        dataset=ts_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2
    )
    return tr_loader, ts_loader
    