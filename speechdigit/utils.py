import os
import random
import numpy as np
import torch
import logging


class TTFS(object):
    def __init__(self, T: int = 8) -> None:
        self.T = T
        self.eps = 1e-3
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.T, *x.shape)
        x = ((1 - x) * (self.T - 1)).floor()
        out.scatter_(0, x.unsqueeze(0).long(), 1)
        return out.to(x)


class Poisson(object):
    def __init__(self, T: int = 8) -> None:
        self.T = T
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(self.T, *x.shape).le(x.unsqueeze(0)).to(x)
    
class Direct(object):
    def __init__(self, T: int = 8) -> None:
        self.T = T
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        repeats = [self.T] + [1] * len(x.shape)
        return x.unsqueeze(dim=0).repeat(repeats)

def accuracy(out, label):
    return float(out.eq(label).sum() / len(out))

def evaluate(model, dataloader, device, out_type='spike'):
    model.eval()
    pred, real = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
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

def TET_loss(outputs, labels, criterion, means=1.0, lamb=1e-3):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total

def seed_all(seed: int = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s-%(filename)s-line:%(lineno)d-%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger