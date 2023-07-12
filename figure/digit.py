import os
import argparse
from matplotlib import pyplot as plt

import torch
import torchvision
import models
import utils
import scipy.io as scio
import projdata

def evaluate(model, dataloader, device, out_type='spike'):
    model.eval()
    pred, real = [], []
    with torch.no_grad():
        for inputs, label in dataloader:
            label = label.long().squeeze(dim=-1)
            out, psp = model(inputs.float().float().to(device))
            if out_type == 'spike':
                pred.extend(out.sum(dim=1).argmax(dim=1))
            elif out_type == 'psp':
                pred.extend(psp.sum(dim=1).argmax(dim=1))
            else:
                raise ValueError('{} not supported.'.format(out_type))
            real.extend(label.to(device))
    model.train()
    return torch.stack(pred), torch.stack(real)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=0.12)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--percent', type=float, default=0.5, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=0)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--encoder', type=str, default='poisson')
    parser.add_argument('--out_type', type=str, default='psp')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nonneg', type=str, default='abs')
    parser.add_argument('--dir_prefix', type=str, default='mnistout')
    parser.add_argument('--data_dir', type=str, default='data')
    
    config = parser.parse_args()
    device = 0

    
    eidrsnn = models.eiDRSNN(
        config.T, config.max_delay, [620, 1024], [1., config.percent, config.percent], 
        config.num_classes, config.sparsity, constraint=config.nonneg
    ).to(device)
    eidrsnn.load_state_dict(torch.load('../sdigitout/eiDRSNN-T100-LR0.05-G0.97-S0.5-P0.5-D0-abs-spike.bin'))
    # rsnn = drsnn
    _, ts_data = projdata.sdigit_dataset('../data/sdigit/', 128, 1)
    
    print()
    
    x, y = list(ts_data)[237]
    print(x, y, x.shape, y.shape)
    print(x, y)
    
    out = eidrsnn(x.float().to(device))
    
    print(out[0].sum(dim=1), out[0].shape)
    # dataset = torchvision.datasets.MNIST('../data/mnist/',train=False, transform=torchvision.transforms.ToTensor(),download=True)
    
    # x, y = dataset[2367]
    
    # plt.imshow(x.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    # plt.savefig(str(y) + '.pdf', )
    
    # _x = utils.Poisson(8)(x.flatten()).unsqueeze(dim=0)
    # out = eidrsnn(_x.to(device))
    
    # print(x.shape, out[0].shape, out[0], out[1])
    scio.savemat('digit-in-out.mat', {'in': x[0].detach().cpu().numpy(), 'out': out[0][0].detach().cpu().numpy()})
    print(scio.loadmat('digit-in-out.mat'))

            
