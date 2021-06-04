import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from scipy import stats


class conv1d_small(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim,8,5,1,2, bias=False)
        self.conv2 = nn.Conv1d(8,16,5,1,2, bias=False)
        self.conv3 = nn.Conv1d(16,8,5,1,2, bias=False)
        self.conv4 = nn.Conv1d(8,1,5,1,2, bias=False)
    def forward(self, x):
        # x i * 1 * n
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        
        return x

class meta_solver_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_local = conv1d_small(2)
        self.conv1d_global = conv1d_small()
    def forward(self, x):
        #x 1 * n * n
        #print(x.shape)
        x = x.squeeze(dim=0)
        n = x.shape[-1]
        x = torch.transpose(x, 0, 1)#n * 1 * n
        global_feature = torch.mean(self.conv1d_global(x), dim=0)# 1 * 1 * n
        x = torch.cat([x,global_feature.repeat(n,1,1)], dim=1)#torch.cat([x, global_feature.repeat([n,1,1])], dim=-1)#n * 1 * n
        x = self.conv1d_local(x)
        output = torch.transpose(x, 0, 1)
        #print(output.shape)
        pi_1 = f.softmax(output.mean(dim=-1), dim=-1)
        return pi_1

class conv1d_large(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim,16,5,1,2, bias=False)
        self.conv2 = nn.Conv1d(16,32,5,1,2, bias=False)
        self.conv3 = nn.Conv1d(32,16,5,1,2, bias=False)
        self.conv4 = nn.Conv1d(16,1,5,1,2, bias=False)
    def forward(self, x):
        # x i * 1 * n
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        
        return x

class meta_solver_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_local = conv1d_large(2)
        self.conv1d_global = conv1d_large()
    def forward(self, x):
        #x 1 * n * n
        #print(x.shape)
        x = x.squeeze(dim=0)
        n = x.shape[-1]
        x = torch.transpose(x, 0, 1)#n * 1 * n
        global_feature = torch.mean(self.conv1d_global(x), dim=0)# 1 * 1 * n
        x = torch.cat([x,global_feature.repeat(n,1,1)], dim=1)#torch.cat([x, global_feature.repeat([n,1,1])], dim=-1)#n * 1 * n
        x = self.conv1d_local(x)
        output = torch.transpose(x, 0, 1)
        #print(output.shape)
        pi_1 = f.softmax(output.mean(dim=-1), dim=-1)
        return pi_1

def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps

def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br