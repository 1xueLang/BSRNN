from typing import List

import torch
import torch.nn as nn
import numpy as np

import snn
from snn import sparse_inner_product, AdaptiveLIF, MultiGaussian, SparseLinearFunc


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
        self.neuron = AdaptiveLIF([self.n_neurons], MultiGaussian.apply)
        self.h_last = 0
        self.register_buffer('neuron_sign', torch.ones(1, n_neurons))
        self.neuron_sign[:, self.n_excitatory:] = -1
        
    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        h = self.synapse_ih(inputs)
        if self.h_last is not 0: h += self.synapse_hh(self.h_last)
        h = nn.functional.dropout(h, p=0.1, training=self.training)
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
        self.neuron = AdaptiveLIF([num_classes], surrogate=MultiGaussian.apply)
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
            h = nn.functional.dropout(h, p=0.1, training=self.training)
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