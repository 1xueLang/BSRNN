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

class sparse_inner_product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: torch.Tensor, 
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask != None: weight = weight.mul(mask)
        ctx.save_for_backward(inputs, weight, mask)
        return inputs.mul(weight.unsqueeze(dim=0))
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> List[Optional[torch.Tensor]]:
        inputs, weight, mask = ctx.saved_tensors
        grad0, grad1, grad2 = None, None, None
        if ctx.needs_input_grad[0]: 
            grad0 = grad_out.mul(weight.unsqueeze(dim=0))
        if ctx.needs_input_grad[1]: 
            grad1 = grad_out.mul(inputs).sum(dim=0)
            if mask != None: grad1 = grad1.mul(mask)
        return grad0, grad1, grad2

class DelayedSynapse(nn.Linear):
    def __init__(self, T: int, t_max: int, in_features: int, out_features: int, 
                 bias: bool = True, sparsity: float = 0.3) -> None:
        super().__init__(in_features, out_features, bias)
        self.T, self.t_max = T, t_max
        self.sparsity = sparsity
        self.register_buffer('mask', self._mask_low_weight(100 * (1 - sparsity)))
        # self.register_buffer('mask', _random_mask(self.sparsity, [out_features, in_features]))
        self.register_buffer('zeros', torch.zeros(1, in_features, out_features))
        self.register_buffer('_delay', torch.randint(0, self.t_max, [in_features, out_features]))
        self.register_buffer('DW', torch.zeros(1, self.t_max, in_features, out_features))
        self._init_delayed_weight()
        self.synapse_func = sparse_inner_product.apply
        self.delayed_input_buffer = []
        self.batch_zeros = None
        self.current = 0
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delayed_input = inputs.unsqueeze(dim=1).unsqueeze(dim=-1) * self.DW
        for i in range(self.t_max):
            if self.current + i >= self.T: break
            ini = self.delayed_input_buffer[self.current + i]
            ini = ini + delayed_input[:, i]
            self.delayed_input_buffer[self.current + i] = ini

        out = self.synapse_func(
            self.delayed_input_buffer[self.current], self.weight.t(), self.mask.t()
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
        
    def _init_delayed_weight(self):
        for i in range(self.t_max):
            self.DW[0][i] = (self._delay == i).float()


class AdaptiveLIF(nn.Module):
    def __init__(self, size: List[int], surrogate: Callable[[torch.Tensor], torch.Tensor],
                 Rm: float = 1., dt: float = 1., b0: float = 0.01, beta: float = 1.8
                 ) -> None:
        super().__init__()
        self.tau_m, self.tau_adp, self.eta, self.u = None, None, 0, 0
        self.surrogate = surrogate
        self.Rm, self.dt, self.b0, self.beta = Rm, dt, b0, beta
        self.init_params(size)
        self.size = size
        self.logs = {
            'spike': [],
            'psp': [],
            'vth': []
        }
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # if self.tau_m is None: self.init_params(inputs)
        self.neuron_charge(inputs=inputs)
        spikes = self.neuron_fire()
        self.update_neuronstat_after_fire(spikes)
        # self.logs['spike'].append(spikes)
        return spikes
        
    def neuron_charge(self, inputs: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(-self.dt / self.tau_m).unsqueeze(dim=0)
        self.u = self.u * alpha + (1 - alpha) * self.Rm * inputs
        # self.logs['psp'].append(self.u)
    
    def neuron_fire(self) -> torch.Tensor:
        v_threshold = self.b0 + self.beta * self.eta
        if isinstance(v_threshold, float):
            v_threshold = torch.ones_like(self.u) * v_threshold
        # self.logs['vth'].append(v_threshold)
        return self.surrogate(self.u - v_threshold)
    
    def update_neuronstat_after_fire(self, spikes: torch.Tensor) -> None:
        self.u = self.u - spikes * (self.eta * self.beta + self.b0)
        # self.u = self.u * (1 - spikes)
        rho = torch.exp(-self.dt / self.tau_adp).unsqueeze(dim=0)
        self.eta = rho * self.eta + (1 - rho) * spikes

    def init_params(self, size: List[int]) -> None:
        self.tau_m = nn.Parameter(torch.FloatTensor(size))
        self.tau_adp = nn.Parameter(torch.FloatTensor(size))
        nn.init.normal_(self.tau_m, 200, 25)
        nn.init.normal_(self.tau_adp, 20, 5)
        
    def detach_history(self) -> None:
        self.eta = self.eta.detach()
        self.u = self.u.detach()
        
    def reset_state(self, batch: torch.Tensor) -> None:
        # self.eta = self.b0
        self.eta = 0
        self.u = 0
        # self.u = torch.rand(batch.shape[0], *self.size).to(batch) * self.b0
     
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
        
        
class recurrent_block(nn.Module):
    def __init__(self, in_size: int, h_size: int, sparsity: float, 
                 surrogate: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.sparse_in = SLinear(in_size, h_size, sparsity=sparsity)
        self.sparse_hh = SLinear(h_size, h_size, sparsity=sparsity)
        self.neuron = AdaptiveLIF([h_size], surrogate=surrogate)
        self.h_last = 0
        nn.init.zeros_(self.sparse_hh.weight)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.sparse_in(inputs)
        if self.h_last is not 0: x = x + self.sparse_hh(self.h_last) 
        self.h_last = self.neuron(x)
        return self.h_last


class RSNN(nn.Module):
    def __init__(self, layers: List[int], num_classes: int, sparsity: float,
                 surrogate: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.sparsity = sparsity
        self.surrogate = surrogate
        self.rsnns = self.make_rnn_layers(layers)
        self.fc_out = SLinear(layers[-1], num_classes, sparsity=sparsity)
        self.neuron = AdaptiveLIF([num_classes], surrogate)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.all_reset(inputs)
        out = []
        for i in range(inputs.shape[1]):
            h = self.rsnns(inputs[:, i])
            out.append(self.neuron(self.fc_out(h)))
        return torch.stack(out, dim=1)
    
    def make_rnn_layers(self, layers: List[int]) -> nn.Sequential:
        rsnns = []
        for i in range(1, len(layers)):
            rsnns.append(
                recurrent_block(layers[i - 1], layers[i], self.sparsity, self.surrogate)
            )
        return nn.Sequential(*rsnns)
    
    def all_reset(self, batch: torch.Tensor) -> None:
        for rnn_layer in self.rsnns:
            rnn_layer.neuron.reset_state(batch)
            rnn_layer.h_last = 0
        self.neuron.reset_state(batch)

class delayed_recurrent_block(nn.Module):
    def __init__(self, T: int, t_max: int, in_size: int, h_size: int, sparsity: float, 
                 surrogate: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.sparse_in = DelayedSynapse(T, t_max, in_size, h_size, sparsity=sparsity)
        self.sparse_hh = DelayedSynapse(T, t_max, h_size, h_size, sparsity=sparsity)
        self.neuron = AdaptiveLIF([h_size], surrogate=surrogate)
        self.h_last = 0
        nn.init.zeros_(self.sparse_hh.weight)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.sparse_in(inputs)
        if self.h_last is not 0: x = x + self.sparse_hh(self.h_last) 
        self.h_last = self.neuron(x)
        return self.h_last

    def reset_state(self, batch: torch.Tensor) -> None:
        self.sparse_in.reset_state(batch)
        self.sparse_hh.reset_state(batch)
        self.h_last = 0
        self.neuron.reset_state(batch)


class DRSNN(nn.Module):
    def __init__(self, T: int, t_max: int, layers: List[int], num_classes: int, sparsity: float,
                 surrogate: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.T = T
        self.t_max = t_max
        self.sparsity = sparsity
        self.surrogate = surrogate
        self.rsnns = self.make_rnn_layers(layers)
        self.fc_out = DelayedSynapse(T, t_max, layers[-1], num_classes, sparsity=sparsity)
        self.neuron = AdaptiveLIF([num_classes], surrogate)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.all_reset(inputs)
        out = []
        for i in range(x.shape[1]):
            h = self.rsnns(inputs[:, i])
            out.append(self.neuron(self.fc_out(h)))
        return torch.stack(out, dim=1)
    
    def make_rnn_layers(self, layers: List[int]) -> nn.Sequential:
        rsnns = []
        for i in range(1, len(layers)):
            rsnns.append(
                delayed_recurrent_block(self.T, self.t_max, layers[i - 1], layers[i], self.sparsity, self.surrogate)
            )
        return nn.Sequential(*rsnns)
    
    def all_reset(self, batch: torch.Tensor) -> None:
        for rnn_layer in self.rsnns:
            rnn_layer.reset_state(batch[:, 0])
        self.fc_out.reset_state(batch[:, 0])
        self.neuron.reset_state(batch[:, 0])

@torch.jit.script
def spike_emiting(potential_cond):
    """
    """
    return potential_cond.ge(0.0).to(potential_cond)


class basic_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, alpha=4.):
        if inputs.requires_grad:
            ctx.save_for_backward(inputs)
            ctx.alpha = alpha
        return spike_emiting(inputs)


class sigmoid(basic_surrogate):
    @staticmethod
    def backward(ctx, grad_out):
        sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
        return grad_out * (1. - sgax) * sgax * ctx.alpha, None


class arc_tan(basic_surrogate):
    @staticmethod
    def backward(ctx, grad_out):
        ret = grad_out * ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)), None
        # print(ctx.saved_tensors[0], ret[0])
        return ret


from typing import List

import torch
import torch.nn as nn
import numpy as np

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
    
