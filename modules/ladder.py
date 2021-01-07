# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:57:08 2020

@author: Andreas
"""

import torch
import torch.nn as nn
import numpy as np


## This should be the layered structure comprised of multiple memory modules
## The idea being that we have a single LSTM for each layer (or other type of 
## memory cell network.) And then we can propagate the output downwards each
## iteration.
## The output of each layer is a vector h of the weight 

## Each layer should have an fs, a c_length and fs[i] should be divisible by
## fs[i-1].


class MLP(nn.Module):
    def __init__(self, q_length, c_length, ms, in_channels = 1):
        super(MLP,self).__init__()
        
        self.q_length = q_length
        self.c_length = c_length
        
        self.fs = ms*ms-1
        
        # Embedding layer used to linearize the input
        self.embedding = nn.ModuleList(
                [nn.Embedding(q_length, c_length) for i in range(self.fs)]
                )
        
        # Fully connected layers used to express correlation between fs samples
        # Assuming that fs is small, this should be an ok way to do things.
        self.lin = nn.Sequential(
                nn.Linear(c_length,in_channels*c_length),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_channels*c_length,q_length),
                nn.LogSoftmax(-1)
                )
        
    def forward(self,f,c,ratio):
        x = torch.zeros([f.size()[0],f.size()[1],self.fs,self.c_length],device = f.device)
        
        for i, l in enumerate(self.embedding):
            x[:,:,i] = l(f[:,:,i])
        
        inp = torch.sum(x,2)
        
        if ratio != 1:
            for i in range(x.size()[1]):
                inp[:,i] = inp[:,i] + c[:,i//ratio]
        else:
            inp = inp + c
        inp = self.lin(inp)
        
        return inp


class Memnode(nn.Module):
    def __init__(self, ms, c_length, layers, mem_type):
        super(Memnode,self).__init__()
        
        self.lin = nn.Linear(ms*ms, c_length, bias=False)
        self.lindrop = nn.Dropout(0.2)
        self.rnn = nn.LSTM(c_length, c_length, num_layers = layers, batch_first = True, dropout = 0.2)
        
        self.c_length = c_length
        
    def forward(self, x, c, state = None):
        x = self.lindrop(self.lin(x))
        ratio = x.size()[1]//c.size()[1]
        if ratio != 1:
            for i in range(x.size()[1]):
                x[:,i] = x[:,i] + c[:,i//ratio]
        else:
            x = x + c
        
        return self.rnn(x, state)
    
class Topnode(nn.Module):
    def __init__(self, ms, c_length, layers, mem_type):
        super(Topnode,self).__init__()
        
        self.rnn = nn.LSTM(ms*ms, c_length, num_layers = layers, batch_first = True, dropout = 0.2)
        
        self.c_length = c_length
        
    def forward(self, x, state = None):
        
        return self.rnn(x, state)
    
class Ladder(nn.Module):
    def __init__(self, ms0, r, c_length, k, q_length, layers, mem_type):
        super(Ladder,self).__init__()
        
        self.ms = np.zeros(k,dtype=np.int)
        self.r = r
        self.c_length = c_length
        self.q_length = q_length
        self.layers = layers
        
        self.k = k
        
        self.ms[0] = ms0
        for i in range(1,k):
            self.ms[i] = self.r[i-1]*self.ms[i-1]

        ## Can clean this up later
        self.rnntop = Topnode(self.ms[-1], self.c_length, self.layers, mem_type)
        self.rnns = nn.ModuleList([Memnode(self.ms[i], self.c_length, self.layers, mem_type) for i in range(1,k-1)])
        self.arnn = MLP(q_length,c_length,ms0)
        self.sector = np.zeros((k-1,2),dtype=np.int)
    
    def update(self, u, si, sj):
        #Figure out which sector we are in at each tier
        
        n_batches = u.size()[0]
        new_sector = [self.ms[-1]*(si//self.ms[-1]), self.ms[-1]*(sj//self.ms[-1])]
        #Can't handle sector[-1] = 0
        if new_sector[1] != self.sector[-1,1]:
            self.sector[-1] = new_sector
            for i in range(self.n_sectors):
                u_view = torch.reshape(
                        u[:,i*self.ms[-1]:(i+1)*self.ms[-1],self.sector[-1,1]:self.sector[-1,1]+self.ms[-1]],
                        [n_batches,1,self.ms[-1]*self.ms[-1]]
                        ).clone()
                _, self.memj[i][-1] = self.rnntop(u_view, self.memj[i][-1])
        if new_sector[0] != self.sector[-1,0]:
            self.sector[-1,0] = new_sector[0]
            u_view = torch.reshape(
                    u[:,self.sector[-1,0]-self.ms[-1]:self.sector[-1,0],self.sector[-1,1]:self.sector[-1,1]+self.ms[-1]],
                    [n_batches,1,self.ms[-1]*self.ms[-1]]
                    ).clone()
            _, self.memi[self.sector[-1,0]//self.ms[-1]][-1] = self.rnntop(u_view, self.memi[self.sector[-1,0]//self.ms[-1]][-1])
        
        
#        for i in range(self.k-2,0,-1):
#            new_sector = self.fs[i]*(s//self.fs[i])
#            if new_sector != self.sector[i-1]:
#                self.sector[i-1] = new_sector
#                u_view = torch.reshape(u[:,self.sector[i-1]-self.fs[i]:self.sector[i-1]],[n_batches,1,self.fs[i]]).clone()
#                _, self.mem[i-1] = self.rnns[i-1](u_view, self.mem[i][0][-1,0].unsqueeze(0).unsqueeze(0), self.mem[i-1])
    
    def forward(self, u, v):
        n_batches = u.size()[0]
        length = u.size()[1]
        #Can be generalized later, but for now this allows only r = [1,1,..,1]
        self.n_sectors = length//self.ms[0]
        
        self.memi = [[(torch.zeros(self.layers,n_batches,self.c_length,requires_grad=False,device=u.device),
                     torch.zeros(self.layers,n_batches,self.c_length,requires_grad=False,device=u.device)) for i in range(self.k-1)] for j in range(self.n_sectors)]
        self.memj = [[(torch.zeros(self.layers,n_batches,self.c_length,requires_grad=False,device=u.device),
                     torch.zeros(self.layers,n_batches,self.c_length,requires_grad=False,device=u.device)) for i in range(self.k-1)] for j in range(self.n_sectors)]
        
        softout = torch.zeros([n_batches,length,length,self.q_length])
        
        v_squares = v.as_strided(
                (n_batches,length-self.ms[0]+1,length-self.ms[0]+1,self.ms[0],self.ms[0]),
                (length*length,length,1,length,1)
                )
        
        margin = self.ms[0]-1
        mask = torch.full((self.ms[0],self.ms[0]),True,dtype = torch.bool)
        mask[margin,margin] = False
        
        for i in range(self.k-1):
            self.sector[i] = 0
        
        for i in range(self.n_sectors-1):
            for j in range(self.n_sectors-1):
                self.update(u,i*self.ms[0],j*self.ms[0])
                for ii in range(self.ms[0]):
                    for jj in range(self.ms[0]):
                        softout[:,i*self.ms[0]+ii+margin+1,j*self.ms[0]+jj+margin+1] = self.arnn(v_squares[:,i*self.ms[0]+ii+1,j*self.ms[0]+jj+1].masked_select(mask).reshape((n_batches,self.ms[0]**2-1)).unsqueeze(0),self.memj[i][0][0]+self.memi[j][0][0],1)[0]
                        v[:,i*self.ms[0]+ii+margin+1,j*self.ms[0]+jj+margin+1] = softout[:,i*self.ms[0]+ii+margin+1,j*self.ms[0]+jj+margin+1].exp().multinomial(1).flatten()
                        u[:,i*self.ms[0]+ii+margin+1,j*self.ms[0]+jj+margin+1] = v[:,i*self.ms[0]+ii+margin+1,j*self.ms[0]+jj+margin+1].float()
        
        return softout, u, v