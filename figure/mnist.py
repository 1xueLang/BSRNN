import os
import argparse
from matplotlib import pyplot as plt

import torch
import torchvision
import models
import utils
import scipy.io as scio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=0.12)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sparsity', type=float, default=0.3)
    parser.add_argument('--percent', type=float, default=0.75, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=0)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--encoder', type=str, default='poisson')
    parser.add_argument('--out_type', type=str, default='psp')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nonneg', type=str, default='abs')
    parser.add_argument('--dir_prefix', type=str, default='mnistout')
    parser.add_argument('--data_dir', type=str, default='data')
    
    config = parser.parse_args()
    device = 0

    
    eidrsnn = models.eiDRSNN(
        config.T, config.max_delay, [784, 256, 256], [1., config.percent, config.percent], 
        config.num_classes, config.sparsity, constraint=config.nonneg
    ).to(device)
    eidrsnn.load_state_dict(torch.load('../mnistout/mnist-eiDRSNN-T8-LR0.1-G0.97-S0.3-P0.75-D0-poisson-psp-abs.bin'))
    # rsnn = drsnn
    dataset = torchvision.datasets.MNIST('../data/mnist/',train=False, transform=torchvision.transforms.ToTensor(),download=True)
    
    x, y = dataset[2367]
    # xl0 = x[0][7].flatten()
    
    
    plt.imshow(xl0.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, 28, 1))
    plt.savefig('new.png')
    
    plt.axis('off')
    plt.imshow(x.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.savefig(str(y) + '.png')
    
    _x = utils.Poisson(8)(x.flatten()).unsqueeze(dim=0)
    out = eidrsnn(_x.to(device))
    
    # print(x.shape, out[0].shape, out[0], out[1])
    # scio.savemat('mnist-in-out.mat', {'in': _x[0].detach().cpu().numpy(), 'out': out[0][0].detach().cpu().numpy()})
    
    # print(scio.loadmat('mnist-in-out.mat'))
    rnn1 = eidrsnn.rsnns[0].neuron
    rnn2 = eidrsnn.rsnns[1].neuron
    lout = eidrsnn.neuron
    
    s_in = _x.detach().numpy()[0]
    s_rnn1 = torch.stack(rnn1.logs['spike'], dim=1).detach().cpu().numpy()[0]
    v_rnn1 = torch.stack(rnn1.logs['vth'], dim=1).detach().cpu().numpy()[0]
    s_rnn2 = torch.stack(rnn2.logs['spike'], dim=1).detach().cpu().numpy()[0]
    v_rnn2 = torch.stack(rnn2.logs['vth'], dim=1).detach().cpu().numpy()[0]
    s_out  = torch.stack(lout.logs['spike'], dim=1).detach().cpu().numpy()[0]
    v_out  = torch.stack(lout.logs['vth'], dim=1).detach().cpu().numpy()[0]
    
    # print(s_in.shape, s_rnn1.shape, s_rnn2.shape, v_out.shape, s_out.shape)
    scio.savemat('spikeinfo.mat', {
        's_in': s_in,
        's_e1': s_rnn1[:, :eidrsnn.rsnns[0].n_excitatory],
        's_i1': s_rnn1[:, eidrsnn.rsnns[0].n_excitatory:],
        'v_e1': v_rnn1[:, :eidrsnn.rsnns[0].n_excitatory],
        'v_i1': v_rnn1[:, eidrsnn.rsnns[0].n_excitatory:],
        's_e2': s_rnn2[:, :eidrsnn.rsnns[1].n_excitatory],
        's_i2': s_rnn2[:, eidrsnn.rsnns[1].n_excitatory:],
        'v_e2': v_rnn2[:, :eidrsnn.rsnns[1].n_excitatory],
        'v_i2': v_rnn2[:, eidrsnn.rsnns[1].n_excitatory:],
        'v_out': v_out,
        's_out': s_out
    })
    # print(scio.loadmat('spikeinfo.mat'))
