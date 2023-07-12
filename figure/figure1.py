import numpy as np
import torch
import torch.nn
import scipy.io as scio

import models

model = models.eiDRSNN(200, 0, [4, 4], [1, 0.75], 1, 1., 'abs')

x = torch.rand([1, 200, 4]).le(0.2).float()

out = model(x)

s_in = x
rnn = model.rsnns[0].neuron
spike = torch.stack(rnn.logs['spike'], dim=1)
psp = torch.stack(rnn.logs['psp'], dim=1)
vth = torch.stack(rnn.logs['vth'], dim=1)
print(spike, psp, vth)

scio.savemat('data200.mat', {
    's_in': s_in.detach().numpy(), # 输入脉冲
    's_e': spike[:,:,:3].detach().numpy(), #兴奋rnn脉冲
    's_i': spike[:,:,-1].detach().numpy(), #抑制rnn脉冲
    'u_e': psp[:,:,:3].detach().numpy(),   #兴奋rnn膜电位
    'u_i': psp[:,:,-1].detach().numpy(),   
    'v_e': vth[:,:,:3].detach().numpy(),   #兴奋rnn阈值
    'v_i': vth[:,:,-1].detach().numpy()
})

# print(out[0], out[1].shape)

# x = scio.loadmat('data100.mat')
# print(x)