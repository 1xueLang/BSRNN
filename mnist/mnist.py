import os
import argparse

import torch

import snn, utils, projdata, eisnn

# 0.2 500 1024 0.3 0.75 -1 L2:1e-3 acc: 98.25
#         512  1                        98.60
#                             5e-3 acc:   
# 0.1 500 128  1              5e-3 acc: 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sparsity', type=float, default=1.)
    parser.add_argument('--percent', type=float, default=0.5, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=-1)
    parser.add_argument('--T', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--encoder', type=str, default='poisson')
    parser.add_argument('--out_type', type=str, default='psp')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nonneg', type=str, default='abs')
    parser.add_argument('--dir_prefix', type=str, default='../mnistout')
    parser.add_argument('--data_dir', type=str, default='../data')
    
    config = parser.parse_args()
    device = 0
    utils.seed_all(1000)
    
    PREFIX = config.dataset + '-eiDRSNN' + '-T' + str(config.T) + '-LR' + str(config.lr) + '-G' + str(config.gamma) + \
        '-S' + str(config.sparsity) + '-P' + str(config.percent) + '-D' + str(config.max_delay) + \
            '-' + config.encoder + '-' + config.out_type + '-' + config.nonneg
    
    print(PREFIX)
    
    logger = utils.get_logger(os.path.join(config.dir_prefix, PREFIX + '.log'))
    # rsnn = snn.RSNN(
    #     [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
    # drsnn = snn.DRSNN(
    #     config.T, config.max_delay, [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
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
    
    if config.dataset == 'mnist':
        tr_data, ts_data = projdata.mnist_dataset(os.path.join(config.data_dir, 'mnist'), config.batch_size, 1024, encoder)
    else:
        tr_data, ts_data = projdata.Fmnist_dataset(os.path.join(config.data_dir, 'Fmnist'), config.batch_size, 1024, encoder)
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
            
    optimizer = torch.optim.AdamW(rsnn.parameters(), lr=config.lr, weight_decay=0)
    # optimizer = torch.optim.SGD(rsnn.parameters(), lr=config.lr, weight_decay=1e-3)
    # optimizer = torch.optim.Adamax(rsnn.parameters(), lr=config.lr, weight_decay=1e-3)
    # optimizer = torch.optim.Adamax([{'params': params3}, {'params': params1, 'lr': lr * 2}, {'params': params2, 'lr': lr * 3}], lr=lr)
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e:1-e/epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
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
            torch.save(rsnn.state_dict(), os.path.join(config.dir_prefix, PREFIX + '.bin'))
        logger.info('e-%d tr-%.2f ts-%.2f best-%.2f' % (e, float(correct / total_n) * 100, acc * 100, best * 100))
        scheduler.step()
            
