import math
from typing import Optional, List, Callable

import torch
import torch.nn as nn
import numpy as np


class SparseLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: torch.Tensor, 
                bias: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        weight = weight * mask if mask is not None else weight
        out = inputs.mm(weight.t())
        ctx.save_for_backward(inputs, weight, mask)
        return out + bias.unsqueeze(dim=0) if bias is not None else out
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        inputs, weight, mask = ctx.saved_tensors
        grad1, grad2, grad3 = [None] * 3
        if ctx.needs_input_grad[0]: grad1 = grad_out.mm(weight)
        if ctx.needs_input_grad[1]:
            grad2 = grad_out.t().contiguous().mm(inputs)
            grad2 = grad2 * mask if mask is not None else grad2
        if ctx.needs_input_grad[2]: grad3 = grad_out.sum(dim=0).squeeze(dim=0)

        return grad1, grad2, grad3, None
   
   
def _random_mask(sparsity: float, size: List[int]) -> torch.Tensor:
    return torch.rand(size).le(sparsity).float()
 
class SLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sparsity: float = 0.3
                 ) -> None:
        super().__init__(in_features, out_features, bias)
        self.sparsity = sparsity
        # self.register_buffer('mask', _random_mask(self.sparsity, [out_features, in_features]))
        self.register_buffer('mask', self._mask_low_weight(100 * (1 - sparsity)))
        self.linear_func = SparseLinearFunc.apply
        # nn.init.constant_(self.bias, 0)
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_func(inputs, self.weight, self.bias, self.mask)
    
    def _mask_low_weight(self, percent: float) -> torch.Tensor:
        pcnt = np.percentile(np.abs(self.weight.detach().numpy()), percent)
        return (self.weight.abs() >= pcnt).float()

     
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) \
        / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class MultiGaussian(torch.autograd.Function):
    Gamma = 0.5
    Lens  = 0.5
    Scale = 6.0
    Hight = 0.15
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return inputs.ge(0.0).to(inputs)
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        _G, _L, _S, _H = \
            MultiGaussian.Gamma, MultiGaussian.Lens, MultiGaussian.Scale, MultiGaussian.Hight
        return grad_out * _G * (gaussian(inputs, 0., _L) * (1. + _H) - \
                gaussian(inputs, _L, _S * _L) * _H - gaussian(inputs, -_L, _S * _L) * _H)
        

class LIF(nn.Module):
    def __init__(self, tau=2., surrogate=MultiGaussian.apply):
        super().__init__()
        self.tau = tau
        self.surrogate = surrogate
        self.u_last = 0
        
    def forward(self, inputs):
        u = self.u_last / self.tau + inputs
        spikes = self.surrogate(u - 1.)
        self.u_last = u * (1 - spikes)
        return spikes
    
    def reset_state(self, batch):
        self.u_last = 0



class AbsNonNeg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return inputs.abs().to(inputs)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        sign = inputs.ge(0.0).float() - inputs.lt(0.0).float()
        return sign * grad_out


class NonNegDelayedSynapse(nn.Linear):
    def __init__(self, T: int, t_max: int, in_features: int, out_features: int, 
                 bias: bool = True, sparsity: float = 0.3, constraint: str = 'abs') -> None:
        super().__init__(in_features, out_features, bias)
        nn.init.orthogonal_(self.weight)
        self.T, self.t_max = T, t_max
        self.sparsity = sparsity
        self.register_buffer('mask', self._mask_low_weight(100 * (1 - sparsity)))
        # self.register_buffer('mask', _random_mask(self.sparsity, [out_features, in_features]))
        self.register_buffer('zeros', torch.zeros(1, in_features, out_features))
        self.register_buffer('delay', torch.randint(0, self.t_max, [in_features, out_features]))
        self.register_buffer('DW', torch.zeros(1, self.t_max, in_features, out_features))
        self._init_delayed_weight()
        self.synapse_func = sparse_inner_product.apply
        self.delayed_input_buffer = []
        self.batch_zeros = None
        self.current = 0
        if constraint == 'abs':
            self.nonneg_constraint = AbsNonNeg.apply
        elif constraint == 'relu':
            self.nonneg_constraint = nn.ReLU()
        elif constraint == 'none':
            self.nonneg_constraint = lambda x: x
        else:
            raise ValueError('{} is not support for non negative constraint.'.format(constraint))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delayed_input = inputs.unsqueeze(dim=1).unsqueeze(dim=-1) * self.DW
        for i in range(self.t_max):
            if self.current + i >= self.T: break
            ini = self.delayed_input_buffer[self.current + i]
            ini = ini + delayed_input[:, i]
            self.delayed_input_buffer[self.current + i] = ini

        out = self.synapse_func(
            self.delayed_input_buffer[self.current], 
            self.nonneg_constraint(self.weight.t()),
            self.mask.t()
        ).sum(dim=1)
        
        self.current += 1
        if self.bias is not None: out = out + self.bias.unsqueeze(dim=0)
        return out
        
    def _mask_low_weight(self, percent: float) -> torch.Tensor:
        pcnt = np.percentile(np.abs(self.weight.detach().numpy()), percent)
        return (self.weight.abs() >= pcnt).float()
    
    def reset_state(self, batch: torch.Tensor) -> None:
        if self.batch_zeros is None or self.batch_zeros.shape[0] != batch.shape[0]:
            self.reset_batch_size(batch=batch)
        self.current = 0
        self.delayed_input_buffer = [self.batch_zeros for i in range(self.T)]

    def reset_batch_size(self, batch: torch.Tensor) -> None:
        self.batch_zeros = self.zeros.repeat(batch.shape[0], 1, 1)
        
    def _init_delayed_weight(self) -> None:
        for i in range(self.t_max):
            self.DW[0][i] = (self.delay == i).float()

class NonNegSLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sparsity: float = 0.3,
                 constraint: str = 'abs') -> None:
        super().__init__(in_features, out_features, bias)
        nn.init.orthogonal_(self.weight)
        self.sparsity = sparsity
        # self.register_buffer('mask', _random_mask(self.sparsity, [out_features, in_features]))
        self.register_buffer('mask', self._mask_low_weight(100 * (1 - sparsity)))
        self.linear_func = SparseLinearFunc.apply
        # nn.init.constant_(self.bias, 0)
        nn.init.xavier_uniform_(self.weight)
        if constraint == 'abs':
            self.nonneg_constraint = AbsNonNeg.apply
        elif constraint == 'relu':
            self.nonneg_constraint = nn.ReLU()
        elif constraint == 'none':
            self.nonneg_constraint = lambda x: x
        else:
            raise ValueError('{} is not support for non negative constraint.'.format(constraint))
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_func(inputs, self.nonneg_constraint(self.weight), self.bias, self.mask)
    
    def _mask_low_weight(self, percent: float) -> torch.Tensor:
        pcnt = np.percentile(np.abs(self.weight.detach().numpy()), percent)
        return (self.weight.abs() >= pcnt).float()
    
    def reset_state(self, batch: torch.Tensor) -> None:
        pass


class ExcitatoryInhibitoryBlock(nn.Module):
    def __init__(self, T: int, t_max: int, n_presynapses: int, n_neurons: int, 
                 percent: float, sparsity: float,  bias: bool = True, constraint: str = 'abs') -> None:
        super().__init__()
        self.n_presynapses = n_presynapses
        self.n_neurons = n_neurons
        self.n_excitatory = int(self.n_neurons * percent)
        self.n_inhibitory = self.n_neurons - self.n_excitatory
        if t_max <= 1:
            self.synapse_ih = NonNegSLinear(n_presynapses, n_neurons, bias, sparsity, constraint)
            self.synapse_hh = NonNegSLinear(n_neurons, n_neurons, bias, sparsity, constraint)
        else:
            self.synapse_ih = NonNegDelayedSynapse(T, t_max, n_presynapses, n_neurons, bias, sparsity, constraint)
            self.synapse_hh = NonNegDelayedSynapse(T, t_max, n_neurons, n_neurons, bias, sparsity, constraint)
        # self.neuron = AdaptiveLIF([self.n_neurons], snn.arc_tan.apply)
        # self.neuron = AdaptiveLIF([self.n_neurons], MultiGaussian.apply)
        self.neuron = LIF()
        self.h_last = 0
        self.register_buffer('neuron_sign', torch.ones(1, n_neurons))
        self.neuron_sign[:, self.n_excitatory:] = -1
        
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        h = self.synapse_ih(inputs)
        if self.h_last is not 0: h += self.synapse_hh(self.h_last)
        spikes = self.neuron(h) * self.neuron_sign
        self.h_last = spikes
        return spikes[:, :self.n_excitatory], spikes[:, self.n_excitatory:]
    
    def reset_state(self, batch: torch.Tensor) -> None:
        self.synapse_ih.reset_state(batch)
        self.synapse_hh.reset_state(batch)
        self.neuron.reset_state(batch)
        self.h_last = 0


class eiDRSNN(nn.Module):
    def __init__(self, T: int, t_max: int, layers: List[int], percent: List[float], 
                 num_classes: int, sparsity: float, constraint: str = 'abs') -> None:
        super().__init__()
        self.T = T
        self.t_max = t_max
        self.sparsity = sparsity
        self.rsnns = self.make_rnn_layers(layers, percent, constraint)
        if t_max <= 1:
            self.fc_out = NonNegSLinear(
                int(layers[-1] * percent[-1]), num_classes, sparsity=sparsity, constraint=constraint
            )
        else:
            self.fc_out = NonNegDelayedSynapse(T, t_max, int(layers[-1] * percent[-1]), 
                                               num_classes, sparsity=sparsity, constraint=constraint
            )
        self.neuron = LIF()
        # self.neuron = AdaptiveLIF([num_classes], surrogate=MultiGaussian.apply)
        # self.neuron = AdaptiveLIF([num_classes], surrogate=snn.arc_tan.apply)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.all_reset(inputs)
        # sum_out = 0
        out, psps = [], []
        for i in range(inputs.shape[1]):
            se, si = self.rsnns[0](inputs[:, i])
            for rnn_layer in self.rsnns[1:]:
                se, si = rnn_layer(se)
            h = self.fc_out(se)
            s = self.neuron(h)
            out.append(s)
            psps.append(h)
            # sum_out += self.neuron(self.fc_out(se))
            # sum_out += self.neuron(self.fc_out(torch.cat([se, si], dim=-1)))
        return torch.stack(out, dim=1), torch.stack(psps, dim=1)
    
    def make_rnn_layers(self, layers: List[int], percent: List[float], constraint: str) -> nn.ModuleList:
        rsnns = []
        for i in range(1, len(layers)):
            rsnns.append(
                ExcitatoryInhibitoryBlock(
                    self.T, self.t_max, int(layers[i - 1] * percent[i - 1]), layers[i], percent[i], self.sparsity, constraint
                )
            )
        return nn.ModuleList(rsnns)
    
    def all_reset(self, batch: torch.Tensor) -> None:
        for rnn_layer in self.rsnns:
            rnn_layer.reset_state(batch[:, 0])
        self.fc_out.reset_state(batch[:, 0])
        self.neuron.reset_state(batch[:, 0])


import os
import argparse

import torch

import utils, projdata

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
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--percent', type=float, default=0.5, help='The proportion of excitatory neurons')
    parser.add_argument('--max_delay', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=11)
    # parser.add_argument('--encoder', type=str, default='poisson')
    parser.add_argument('--out_type', type=str, default='spike')
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--nonneg', type=str, default='abs')
    parser.add_argument('--dir_prefix', type=str, default='../sdigitout')
    parser.add_argument('--data_dir', type=str, default='../data')
    
    config = parser.parse_args()
    device = 0
    # utils.seed_all(1000)
    
    PREFIX = 'LIF' + '-T' + str(100) + '-LR' + str(config.lr) + '-G' + str(config.gamma) + \
        '-S' + str(config.sparsity) + '-P' + str(config.percent) + '-D' + str(config.max_delay) + '-' + config.nonneg + '-' + config.out_type
    
    print(PREFIX)
    
    logger = utils.get_logger(os.path.join(config.dir_prefix, PREFIX + '.log'))
    # rsnn = snn.RSNN(
    #     [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
    # drsnn = snn.DRSNN(
    #     config.T, config.max_delay, [784, 256, 256], config.num_classes, config.sparsity, snn.MultiGaussian.apply
    # ).to(device)
    
    eidrsnn = eiDRSNN(
        100, config.max_delay, [620, 1024], [1., config.percent, config.percent], 
        config.num_classes, config.sparsity, constraint=config.nonneg
        ).to(device)
    
    # rsnn = drsnn
    rsnn = eidrsnn
    
    
    tr_data, ts_data = projdata.sdigit_dataset(os.path.join(config.data_dir, 'sdigit'), config.batch_size, 256)
    
    params1 = []
    params2 = []
    params3 = []
    
    for n, p in rsnn.named_parameters():
        if 'tau_m' in n:
            params1.append(p)
        elif 'tau_adp' in n:
            params2.append(p)
        else:
            params3.append(p)
            
    # optimizer = torch.optim.Adam(rsnn.parameters(), lr=config.lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(rsnn.parameters(), lr=config.lr, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(rsnn.parameters(), lr=config.lr, weight_decay=1e-4)
    # optimizer = torch.optim.Adamax([{'params': params3}, {'params': params1, 'lr': config.lr * 2}, {'params': params2, 'lr': config.lr * 3}], lr=config.lr)
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e:1-e/epoch)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    best = 0.0
    for e in range(config.num_epochs):
        correct = 0.
        total_n = 0.
        for i, (x, y) in enumerate(tr_data):
            y = y.long().squeeze(dim=-1)
            out, psps = rsnn(x.float().to(device))
            # print(out.sum())
            one_hot_label = torch.nn.functional.one_hot(y, config.num_classes)
            
            if config.out_type == 'spike':
                loss = torch.nn.functional.cross_entropy(out.sum(dim=1), y.to(device))
                # loss = torch.square(out.mean(dim=1) - one_hot_label.to(device)).sum() / 2
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
        acc = float(utils.accuracy(*evaluate(rsnn, ts_data, device, config.out_type)))
        if acc > best: 
            best = acc
            torch.save(rsnn.state_dict(), os.path.join(config.dir_prefix, PREFIX + '.bin'))
        logger.info('e-%d tr-%.2f ts-%.2f best-%.2f' % (e, float(correct / total_n) * 100, acc * 100, best * 100))
        scheduler.step()
            
