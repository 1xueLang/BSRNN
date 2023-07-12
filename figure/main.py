import os
import argparse

import torch

import utils, projdata
import models as eisnn
import scipy.io as scio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sparsity', type=float, default=0.3)
    parser.add_argument('--percent', type=float, default=0.5, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=0)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--encoder', type=str, default='poisson')
    parser.add_argument('--out_type', type=str, default='psp')
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--nonneg', type=str, default='none')
    parser.add_argument('--dir_prefix', type=str, default='mnistout')
    parser.add_argument('--data_dir', type=str, default='../data')
    
    config = parser.parse_args()
    device = 0
    utils.seed_all(1000)
    
    eidrsnn = eisnn.eiDRSNN(
        config.T, config.max_delay, [784, 256, 256], [1., config.percent, config.percent], 
        config.num_classes, config.sparsity, constraint=config.nonneg
        ).to(device)
    
    # rsnn = drsnn
    rsnn = eidrsnn
    rsnn.train()
    
    if config.encoder == 'direct':
        encoder = utils.Direct(T=config.T)
    elif config.encoder == 'poisson':
        encoder = utils.Poisson(T=config.T)
    elif config.encoder == 'ttfs':
        encoder = utils.TTFS(T=config.T)
    else:
        raise ValueError('{} is not supported.'.format(config.encoder))
    
    tr_data, ts_data = projdata.mnist_dataset(os.path.join(config.data_dir, 'mnist'), config.batch_size, 1024, encoder)
    
    
    rnn1_weight = rsnn.rsnns[1].synapse_hh.weight
    rnn1_weight_mask = rsnn.rsnns[1].synapse_hh.mask
    rnn1_sign = rsnn.rsnns[1].neuron_sign
    # print(rnn1_sign)
    saved_weight = (rnn1_weight * rnn1_weight_mask) * rnn1_sign
    
    scio.savemat('none_weight_before.mat', {'w': saved_weight.cpu().detach().numpy()})
    # w = scio.loadmat('signed_weight_before.mat')
    # print(w['w'][0])
    # params1 = []
    # params2 = []
    # params3 = []
    
    # for n, p in rsnn.named_parameters():
    #     if 'tau_m' in n:
    #         params1.append(p)
    #     elif 'tau_adp' in n:
    #         params2.append(p)
    #     else:
    #         params3.append(p)
            
    optimizer = torch.optim.AdamW(rsnn.parameters(), lr=config.lr)
    # optimizer = torch.optim.Adamax(rsnn.parameters(), lr=lr)
    # optimizer = torch.optim.Adamax([{'params': params3}, {'params': params1, 'lr': lr * 2}, {'params': params2, 'lr': lr * 3}], lr=lr)
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e:1-e/epoch)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    best = 0.0
    for e in range(config.num_epochs):
        correct = 0.
        total_n = 0.
        for i, (x, y) in enumerate(tr_data):
            out, psps = rsnn(x.to(device))
            one_hot_label = torch.nn.functional.one_hot(y, 10)
            
            if config.out_type == 'spike':
                loss = torch.nn.functional.cross_entropy(out.sum(dim=1), y.to(device))
                correct += out.sum(dim=1).argmax(dim=1).eq(y.to(device)).sum()
                total_n += out.shape[0]
            elif config.out_type == 'psp':
                loss = utils.TET_loss(psps, y.to(device), torch.nn.functional.cross_entropy)
                correct += psps.sum(dim=1).argmax(dim=1).eq(y.to(device)).sum()
                total_n += psps.shape[0]
            else:
                raise ValueError('{} is not supported.'.format(config.out_type))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rsnn.parameters(), 3., 2)
            optimizer.step()
            
            if (1 + i) % 100 == 0:
                print('tr acc {:.2f}, Loss {:.5f}'.format(100 * float(correct / total_n), float(loss)))
        acc = float(utils.accuracy(*utils.evaluate(rsnn, ts_data, device, config.out_type)))
        if acc > best: 
            best = acc
            torch.save(rsnn.state_dict(), 'mnist.bin')
            saved_weight = (rnn1_weight * rnn1_weight_mask) * rnn1_sign
            scio.savemat('none_weight_after.mat', {'w': saved_weight.cpu().detach().numpy()})
            
        print('e-%d tr-%.2f ts-%.2f best-%.2f' % (e, float(correct / total_n) * 100, acc * 100, best * 100))
        scheduler.step()
            
