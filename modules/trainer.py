import torch
import torch.nn as nn
from torch import optim
import numpy as np

import modules.ladder as ladder


if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

class Trainer(nn.Module):
    def __init__(self,mask_length, r, c_length, k, q_length, layers, beta, seq_length, n_batches):
        super(Trainer,self).__init__()
        
        self.u = torch.ones([n_batches,seq_length,seq_length],device = dev,dtype = torch.float)
        self.v = torch.ones([n_batches,seq_length,seq_length],device = dev,dtype = torch.long)
        
        self.net = ladder.Ladder(mask_length, r, c_length, k, q_length, layers, "lstm").to(dev)
        self.optimizer = optim.Adam(self.net.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8)
        
        self.c_length = c_length
        self.mask_length = mask_length
        self.k = k
        self.r = r
        self.beta = beta
        self.seq_length = seq_length
        self.n_batches = n_batches
        
    def anneal(self,beta):
        self.beta = beta
    
    def free_energy(self,res,v,beta,j_exchange=1):
        batches = res.size()[0]
        sequence = res.size()[1]
        
        r = torch.zeros(batches,requires_grad=False,device=v.device)
        e = torch.zeros(batches,requires_grad=False,device=v.device)
        qi = torch.zeros([batches, sequence, sequence],device=v.device)
        for s in range(batches):
            for i in range(sequence):
                for j in range(sequence):
                    qi[s,i,j] = res[s,i,j][v[s,i,j]]
        
        logprob = qi.sum((1,2))
        
        with torch.no_grad():
        
            for s in range(batches):
                for i in range(sequence):
                    e[s] = e[s] + torch.sum((v[s,i-1,:]*2-1) * (v[s,i,:]*2-1))
                    e[s] = e[s] + torch.sum((v[s,:,i-1]*2-1) * (v[s,:,i]*2-1))
                r[s] = e[s]*j_exchange*beta + logprob[s]
        loss_reinforce = torch.mean((r-r.mean())*logprob)
        
        
        return loss_reinforce, r.mean()/(beta*sequence*sequence)
    
    def heat_capacity(self,res,v,beta,j_exchange=1):
        
        
        batches = res.size()[0]
        sequence = res.size()[1]
        
        e = torch.zeros(batches,requires_grad=False,device=v.device)

        with torch.no_grad():
        
            for s in range(batches):
                for i in range(sequence):
                    e[s] = e[s] + torch.sum((v[s,i-1,:]*2-1) * (v[s,i,:]*2-1))
                    e[s] = e[s] + torch.sum((v[s,:,i-1]*2-1) * (v[s,:,i]*2-1))
                e[s] = e[s]*j_exchange
        
        
        return (beta**2)*e.var()/(sequence**2)
    
    def magnetization(self,v):
        return (v*2-1).type(torch.float).mean()
    
    def abs_magnetization(self,v):
        return (v*2-1).type(torch.float).mean((1,2)).abs().mean()
    
    def step(self,u,v):
        self.optimizer.zero_grad()
        res, u, v = self.net(u,v)
        loss, fm = self.free_energy(res[:,self.mask_length:,self.mask_length:],v[:,self.mask_length:,self.mask_length:],self.beta)
        hc = self.heat_capacity(res[:,self.mask_length:,self.mask_length:],v[:,self.mask_length:,self.mask_length:],self.beta)
        m = self.magnetization(v[:,self.mask_length:,self.mask_length:])
        abs_m = self.abs_magnetization(v[:,self.mask_length:,self.mask_length:])
        loss.backward()
        self.optimizer.step()
        
        return fm, hc, m, abs_m
        
    def train(self,epochs,minibatches,beta,prune_every=20):
        self.f_array = torch.zeros(epochs,requires_grad=False)
        self.hc_array = torch.zeros(epochs,requires_grad=False)
        self.m_array = torch.zeros(epochs,requires_grad=False)
        self.abs_m_array = torch.zeros(epochs,requires_grad=False)
        
        for i in range(epochs):
            permutation = torch.randperm(self.n_batches)
            for j in range(0,self.n_batches+1-minibatches,minibatches):
                indices = permutation[j:j+minibatches]
                u_minibatch = self.u[indices]
                v_minibatch = self.v[indices]
                f, hc, m, abs_m = self.step(u_minibatch,v_minibatch)
                with torch.no_grad():
                    self.f_array[i] = f
                    self.hc_array[i] = hc
                    self.m_array[i] = m
                    self.abs_m_array[i] = abs_m
                print('Now in epoch ',i)
                print('Free energy pr. spin is ',f.item())
                print('Heat capacity pr. spin is ',hc.item())
                if np.mod(i+1,prune_every) == 0:
                    print('Now pruning')
        
        
        
        return self.f_array, self.hc_array, self.m_array, self.abs_m_array