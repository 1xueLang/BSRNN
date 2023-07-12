import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR,LambdaLR
import math
import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import scipy.io as scio
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="choose the task: smnist and psmnist", type=str,default="psmnist")
parser.add_argument("--ec_f", help="choose the encode function: rbf, rbf-lc, poisson", type=str,default='rbf')
parser.add_argument("--dc_f", help="choose the decode function: adp-mem, adp-spike, integrator", type=str,default='adp-spike')#'integrator')
parser.add_argument("--batch_size", help="set the batch_size", type=int,default=64)
parser.add_argument("--encoder", help="set the number of encoder", type=int,default=80)
parser.add_argument("--num_epochs", help="set the number of epoch", type=int,default=200)
parser.add_argument("--learning_rate", help="set the learning rate", type=float,default=8e-3)
parser.add_argument("--len", help="set the length of the gaussian", type=float,default=0.5)
parser.add_argument('--network', nargs='+', type=int,default=[620,256,256])

'''
加载数据
'''
torch.manual_seed(0)
def load_dataset(task='smnist'):
    if task == 'smnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif task == 'psmnist':
        X_train = np.load('./ps_data/ps_X_train.npy')
        X_test = np.load('./ps_data/ps_X_test.npy')
        y_train = np.load('./ps_data/Y_train.npy')
        y_test = np.load('./ps_data/Y_test.npy')
    else:
        print('only two task, -- smnist and psmnist')
        return 0
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = data.TensorDataset(X_train,y_train) # create train datset
    test_dataset = data.TensorDataset(X_test,y_test) # create test datset

    return train_dataset,test_dataset



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


'''
RLIF 模型
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5

'''
高斯函数
'''
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

'''
计算点火与否：膜电压-阈值
定义这块的反传函数
'''
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # 传入输入：input = membrane potential- threshold，默认调这个
        ctx.save_for_backward(input) #所有要用到的反传参数都得存这里面
        return input.gt(0).float()  # is firing ??? 大于0为1，

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients，近似梯度，反传再调
        input, = ctx.saved_tensors   # input = membrane potential- threshold
        grad_input = grad_output.clone() #输出脉冲总和
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15 #论文中的h
        #temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        # temp = F.relu(1-input.abs()) 93.53
        # temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply


'''
ALIF模型，tau_adp tau_m 可调节
'''
def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    #     tau_adp = torch.FloatTensor([tau_adp])
    # alpha = torch.exp(-1. * dt / tau_m).cuda()
    # ro = torch.exp(-1. * dt / tau_adp).cuda()
    alpha = torch.exp(-1. * dt / tau_m)
    ro = torch.exp(-1. * dt / tau_adp) #rho：ρ
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike #eta：η
    B = b_j0 + beta * b           #theta：θ

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    # spike = F.relu(inputs_)
    return mem, spike, B, b

'''LIF模型不发出脉冲 '''
def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    # alpha = torch.exp(-1. * dt / tau_m).cuda()
    alpha = torch.exp(-1. * dt / tau_m)
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

'''
STEP 3b: CREATE MODEL CLASS
'''
class RNN_custom(nn.Module):
    def __init__(self, input_size, stride, hidden_dims, output_size,DC_f='mem'):
        super(RNN_custom, self).__init__()

        self.DC_f = DC_f #ALIF
        self.b_j0 = b_j0 #0.01
        self.stride = stride#时间步1
        self.input_size = input_size #输入神经元4
        self.output_size = output_size #输入神经元10
        self.criterion = nn.CrossEntropyLoss() #交叉熵

        '''
        1输入+两层RNN隐藏层+1dense+1输出的结构
        '''
        self.r1_dim = hidden_dims[0] #64
        self.r2_dim = hidden_dims[1] #256
        self.d1_dim = hidden_dims[2] #256
        self.i2h = nn.Linear(input_size, self.r1_dim)
        self.h2h = nn.Linear(self.r1_dim, self.r1_dim)
        self.h2d = nn.Linear(self.r1_dim, self.r2_dim)
        self.d2d = nn.Linear(self.r2_dim, self.r2_dim)
        self.dense1 = nn.Linear(self.r2_dim, self.d1_dim)
        self.d2o = nn.Linear(self.d1_dim, self.output_size)


        '''
        ALIF参数设置 tau_adp 和 tau_m
        '''
        self.tau_adp_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        self.tau_m_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_m_o = nn.Parameter(torch.Tensor(self.output_size))

        '''
        初始化权重
        '''
        # nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)  #跨层均匀分布
        nn.init.zeros_(self.h2h.weight)           #同层内权重初始化为0
        nn.init.xavier_uniform_(self.h2d.weight)  #跨层均匀分布
        # nn.init.orthogonal_(self.d2d.weight)
        nn.init.zeros_(self.d2d.weight)           #同层内权重初始化为0
        nn.init.xavier_uniform_(self.dense1.weight)#跨层均匀分布
        nn.init.xavier_uniform_(self.d2o.weight)#跨层均匀分布

        '''
        偏置b初始化为0
        '''
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        # nn.init.normal_(self.tau_adp_r1, 700,25)
        # nn.init.normal_(self.tau_adp_r2, 700,25)
        # nn.init.normal_(self.tau_adp_o, 700,25)
        # nn.init.normal_(self.tau_adp_d1, 700,25)
        '''
        参数初始化
        给定均值和标准差的正态分布
        '''
        nn.init.normal_(self.tau_adp_r1, 200,25) #均值200，方差25的正态分布
        nn.init.normal_(self.tau_adp_r2, 200,25)
        nn.init.normal_(self.tau_adp_o, 200,25)
        nn.init.normal_(self.tau_adp_d1, 200,25)

        nn.init.normal_(self.tau_m_r1, 20,5)
        nn.init.normal_(self.tau_m_r2, 20,5)
        nn.init.normal_(self.tau_m_o, 20,5)
        nn.init.normal_(self.tau_m_d1, 20,5)

        self.b_r1 =self.b_r2 = self.b_o  = self.b_d1  = 0
    
    def compute_input_steps(self,seq_num):
        return int(seq_num/self.stride)

    ''' 前向传播
        调RNN_custom时自动调这个
    '''
    def forward(self, input,labels,tbptt_steps=50,Training=True,optimizer=None):
        batch_size, seq_num, input_dim = input.shape
        #每层的阈值基础 每层的阈值都是b_jo 0.01
        self.b_r1 =self.b_r2 = self.b_o  = self.b_d1  = torch.tensor(b_j0)

        # 初始化每层的膜电压和输出
        self.initial_neuron_states(batch_size)

        #初始化一些参数
        loss = 0
        l1_spikes = []
        l2_spikes = []
        l3_spikes = []
        out_spikes = []
        # input = (input/255.).gt(0.1).float() #大于0.1设置为1，否则为0
        # input_steps  = self.compute_input_steps(seq_num)#时间步等于序列长度/设置的步长
        input_steps = 100

        for i in range(input_steps): #遍历时间步
            # start_idx = i*self.stride
            # if start_idx < (seq_num - self.input_size):
            #     input_x = input[:, start_idx:start_idx+self.input_size, :].reshape(-1,self.input_size) #每个时间步i喂入i：i+4的元素 300*4的输入
            # else:
            #     input_x = input[:, -self.input_size:, :].reshape(-1,self.input_size)
            input_x = input[:, i].to(device)
            #print(input_x.shape)
            # RNN1
            # print(input_x.get_device(), self.r1_spike.get_device())
            h_input = self.i2h(input_x.float()) + self.h2h(self.r1_spike)
            self.r1_mem, self.r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input,self.r1_mem, self.r1_spike, self.tau_adp_r1, self.tau_m_r1,self.b_r1)
            # RNN2
            d_input = self.h2d(self.r1_spike) + self.d2d(self.r2_spike)
            self.r2_mem, self.r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, self.r2_mem, self.r2_spike, self.tau_adp_r2,self.tau_m_r2, self.b_r2)
            # dense
            self.d1_mem, self.d1_spike, theta_d1, self.b_d1 = mem_update_adp(self.dense1(self.r2_spike), self.d1_mem, self.d1_spike, self.tau_adp_d1,self.tau_m_d1, self.b_d1)


            #output 层
            o_input = self.d2o(self.d1_spike)
            if self.DC_f[:3]=='adp':
                self.d2o_mem, self.d2o_spike, theta_o, self.b_o = mem_update_adp(o_input,self.d2o_mem, self.d2o_spike, self.tau_adp_o, self.tau_m_o, self.b_o)
                
            elif self.DC_f == 'integrator':
                self.d2o_mem = output_Neuron(o_input,self.d2o_mem, self.tau_m_o)
                
            
            
            if i > 0: 
                if self.DC_f == 'adp-mem':    #输出是整个训练周期膜电压概率总和
                    self.output_sumspike = self.output_sumspike + F.softmax(self.d2o_mem,dim=1)
                elif self.DC_f =='adp-spike': #输出是整个训练周期的脉冲个数总和
                    self.output_sumspike = self.output_sumspike + self.d2o_spike
                elif self.DC_f =='integrator':#输出是整个训练周期膜电压概率总和
                    self.output_sumspike =self.output_sumspike+ F.softmax(self.d2o_mem,dim=1)

                if Training and i % tbptt_steps==0: #每隔100个step更新一次权重
                    loss = self.criterion(self.output_sumspike,labels) #计算loss
                    # print(i,loss)
                    loss.backward(retain_graph=True) #计算所有参数变化量
                    optimizer.step()                #更新参数变化量
                    #保存参数
                    save_spike = (self.r1_spike.detach(),self.r2_spike.detach(),self.d1_spike.detach(),self.d2o_spike.detach(),self.output_sumspike.detach())
                    save_threshold = (self.b_r1.detach(),self.b_r2.detach(),self.b_d1.detach(),self.b_o.detach())
                    save_mem = (self.r1_mem.detach(),self.r2_mem.detach(),self.d1_mem.detach(),self.d2o_mem.detach())

                    self.reset_neuron_states(save_mem,save_spike,save_threshold)

                    optimizer.zero_grad()           #参数变化量置0

            l1_spikes.append(self.r1_spike.detach().cpu().numpy())
            l2_spikes.append(self.r2_spike.detach().cpu().numpy())
            l3_spikes.append(self.d1_spike.detach().cpu().numpy())
            out_spikes.append(self.d2o_spike.detach().cpu().numpy())

        return self.output_sumspike,  [l1_spikes,l2_spikes,l3_spikes,out_spikes]

    '''
    初始化2个RNN层，1dense层，1输出层的膜电压 0-0.01和脉冲
    前三层脉冲和膜电压一样
    输出层膜电压0-0。01，输出全0，输出总和全0
    '''
    def initial_neuron_states(self,batch_size):
        # self.r1_mem = self.r1_spike = torch.rand(batch_size, self.r1_dim).cuda()*self.b_j0
        # self.r2_mem = self.r2_spike = torch.rand(batch_size, self.r2_dim).cuda()*self.b_j0
        # self.d1_mem = self.d1_spike = torch.rand(batch_size, self.d1_dim).cuda()*self.b_j0
        # self.d2o_mem = torch.rand(batch_size, output_dim).cuda()*self.b_j0
        # self.d2o_spike = self.output_sumspike = torch.zeros(batch_size, output_dim).cuda()
        self.r1_mem = self.r1_spike = torch.rand(batch_size, self.r1_dim).to(device) * self.b_j0
        self.r2_mem = self.r2_spike = torch.rand(batch_size, self.r2_dim).to(device) * self.b_j0
        self.d1_mem = self.d1_spike = torch.rand(batch_size, self.d1_dim).to(device) * self.b_j0
        self.d2o_mem = torch.rand(batch_size, output_dim).to(device) * self.b_j0
        self.d2o_spike = self.output_sumspike = torch.zeros(batch_size, output_dim).to(device)

    def reset_neuron_states(self,saved_mem,saved_spike,saved_threshold): 
        self.r1_mem,self.r2_mem,self.d1_mem,self.d2o_mem =saved_mem
        
        self.b_r1,self.b_r2,self.b_d1,self.b_o = saved_threshold
        self.r1_spike,self.r2_spike,self.d1_spike,self.d2o_spike,self.output_sumspike = saved_spike



def train(model, num_epochs,train_loader,test_loader,file_name,MyFile):
    acc = []
    
    best_accuracy = 80
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            # images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)# .view reshape图片，.requires_grad_()梯度数据可以被保存
            labels = labels.squeeze(dim=1).long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _ = model(images,labels,tbptt_steps=100,Training=True,optimizer=optimizer)
            # Calculate Loss: softmax --> cross entropy loss
            loss = model.criterion(outputs, labels) #softmax+CE+onehot
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1) #batch_size个样本对应的最多点火数为预测标签
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.long().cpu()).sum()
            else:
                correct += (predicted == labels).sum()
            accuracy = 100. * correct.numpy() / total
        scheduler.step()
        # accuracy = test(model, train_loader)
        ts_acc = test(model,test_loader)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '_'+file_name+'-multi_input.pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        res_str = 'epoch: '+str(epoch)+' Loss: '+ str(loss.item())+'. Tr Accuracy: '+ str(accuracy)+ '. Ts Accuracy: '+str(ts_acc) + '. best: ' + str(best_accuracy)
        print(res_str)
        MyFile.write(res_str)
        MyFile.write('\n')
    return acc


def test1(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, states = model(images,labels,Training=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy

def test(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    fr1 = []
    fr2 = []
    fr3 = []
    fr4 = []
    for images, labels in dataloader:
        # images = images.view(-1, seq_dim, input_dim).to(device)
        images = images.to(device)
        labels = labels.squeeze(dim=1)

        outputs, states = model(images,labels,Training=False)
        r1_spike_np = np.array(states[0])
        r2_spike_np = np.array(states[1])
        d1_spike_np = np.array(states[2])
        d2_spike_np = np.array(states[3])
        fr1.append(np.mean(d1_spike_np))
        fr2.append(np.mean(r1_spike_np))
        fr3.append(np.mean(r2_spike_np))
        fr4.append(np.mean(d2_spike_np))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    print('fr: ',np.mean(fr1),np.mean(fr2),np.mean(fr3),np.mean(fr4))
    accuracy = 100. * correct.numpy() / total
    return accuracy

def predict(model,test_loader):
    # Iterate through test dataset
    result = np.zeros(1)
    for images, labels in test_loader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _,_,_ = model(images)
        # _, Predicted = torch.max(outputs.data, 1)
        # result.append(Predicted.data.cpu().numpy())
        predicted_vec = outputs.data.cpu().numpy()
        Predicted = predicted_vec.argmax(axis=1)
        result = np.append(result,Predicted)
    return np.array(result[1:]).flatten()

if __name__ == '__main__':
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    task = args.task
    EC_f = args.ec_f
    DC_f = args.dc_f
    num_encode=args.encoder

    # train_dataset,test_dataset = load_dataset(task)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
    train_loader, test_loader = sdigit_dataset('../../../../data/sdigit/', batch_size, 256)


    input_dim = 1
    input_size=620
    stride = 1
    hidden_dims = args.network#[64,256,256]
    output_dim = 11
    seq_dim = int(784 / input_dim)  # Number of steps to unroll

    #初始化RNN
    model = RNN_custom(input_size, stride,hidden_dims, output_dim,DC_f=DC_f) #RNN网络初始化 调用__init__

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model.to(device)

    # accuracy = test(model,test_loader)
    # print('test Accuracy: ', accuracy)

    criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if EC_f == 'rbf-lc':
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.d2o.weight, model.d2o.bias,model.threshold_event]
    else:
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.dense1.weight, model.dense1.bias, 
                model.d2o.weight, model.d2o.bias]

    optimizer = torch.optim.Adamax([
        {'params': base_params},
        {'params': model.tau_adp_r1, 'lr': learning_rate * 3},
        {'params': model.tau_adp_r2, 'lr': learning_rate * 3},
        {'params': model.tau_adp_d1, 'lr': learning_rate * 3},
        {'params': model.tau_adp_o, 'lr': learning_rate * 3},
        {'params': model.tau_m_r1, 'lr': learning_rate * 2},
        {'params': model.tau_m_r2, 'lr': learning_rate * 2},
        {'params': model.tau_m_d1, 'lr': learning_rate * 2},
        {'params': model.tau_m_o, 'lr': learning_rate * 2},
        ],
        lr=learning_rate, weight_decay=5e-4)


    # scheduler = StepLR(optimizer, step_size=25, gamma=.75)
    # scheduler = MultiStepLR(optimizer, milestones=[25,50,100,150],gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[50,100,150],gamma=0.5)
    scheduler = LambdaLR(optimizer,lr_lambda=lambda epoch:1-epoch/200) # new lr = lr*(1-epoch/200)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print('Time: ',dt_string)
    file_name = 'Task-'+task+'||Time-'+ dt_string+'||EC_f--'+EC_f+'||DC_f--'+DC_f+'||multiinput'
    MyFile=open('./result_file/'+file_name+'.txt','w')
    MyFile.write(file_name)
    MyFile.write('\nnetwork: ['+str(hidden_dims[0])+' '+str(hidden_dims[1])+']')
    MyFile.write('\nlearning_rate: '+str(learning_rate))
    MyFile.write('\nbatch_size: '+str(batch_size))
    MyFile.write('\n\n =========== Result ======== \n')
    acc = train(model, num_epochs,train_loader,test_loader,file_name,MyFile)
    accuracy = test(model,test_loader)
    print('test Accuracy: ', accuracy)
    MyFile.write('test Accuracy: '+ str(accuracy))
    MyFile.close()

    ###################
    ##  Accuracy  curve
    ###################
    if num_epochs >= 10:
        plt.plot(acc)
        plt.title('Learning Curve -- Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy: %')
        plt.show()

