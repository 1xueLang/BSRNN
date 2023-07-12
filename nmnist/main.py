import os
import argparse

import torch

import snn, utils, projdata, eisnn

def preprocess(imgs):
    return imgs.flatten(2)

def evaluate(model, dataloader, device, out_type='spike'):
    model.eval()
    pred, real = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            inputs = preprocess(inputs.float().to(device))
            out, psp = model(inputs)
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
    parser.add_argument('--dataset', type=str, default='nmnist')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--sparsity', type=float, default=1)
    parser.add_argument('--percent', type=float, default=0.3, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=-1)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--out_type', type=str, default='spike')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nonneg', type=str, default='abs')
    parser.add_argument('--dir_prefix', type=str, default='./nodp')
    parser.add_argument('--data_dir', type=str, default='../nmnistdata')
    
    config = parser.parse_args()
    device = 0
    utils.seed_all(3704)
    
    PREFIX = config.dataset + '-eiDRSNN' + '-T' + str(config.T) + '-LR' + str(config.lr) + '-G' + str(config.gamma) + \
        '-S' + str(config.sparsity) + '-P' + str(config.percent) + '-D' + str(config.max_delay) + \
            '-' + config.out_type + '-' + config.nonneg
    
    print(PREFIX)
    
    logger = utils.get_logger(os.path.join(config.dir_prefix, PREFIX + '.log'))
    # rsnn = snn.RSNN(
    #     [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
    # drsnn = snn.DRSNN(
    #     config.T, config.max_delay, [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
    eidrsnn = eisnn.eiDRSNN(
        config.T, config.max_delay, [2312, 512,], [1., config.percent, config.percent], 
        config.num_classes, config.sparsity, constraint=config.nonneg
    ).to(device)
    
    # rsnn = drsnn
    rsnn = eidrsnn
    rsnn.train()
    
    tr_data, ts_data = projdata.n_mnist_dataset(config.data_dir, config.T, config.batch_size, 1024)
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)
    
    best = 0.0
    for e in range(config.num_epochs):
        correct = 0.
        total_n = 0.
        for i, (x, y) in enumerate(tr_data):
            x = preprocess(x.float().to(device))
            out, psps = rsnn(x)
            one_hot_label = torch.nn.functional.one_hot(y, config.num_classes)
            
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
            # torch.nn.utils.clip_grad_norm_(rsnn.parameters(), 3., 2)
            optimizer.step()
            
            if (1 + i) % 100 == 0:
                print('tr acc {:.2f}, Loss {:.5f}'.format(100 * float(correct / total_n), float(loss)))
        acc = float(utils.accuracy(*evaluate(rsnn, ts_data, device, config.out_type)))
        if acc > best: 
            best = acc
            torch.save(rsnn.state_dict(), os.path.join(config.dir_prefix, PREFIX + '.bin'))
        logger.info('e-%d tr-%.2f ts-%.2f best-%.2f' % (e, float(correct / total_n) * 100, acc * 100, best * 100))
        scheduler.step()
            
