"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We add the code in inner_SGD for the Mixed Gradient Preprocessing and Memory Regularization in our ARWarpGrad.
"""

import math
import torch
import numpy as np
from torch import optim


W= 10
pg_memory=[]
new_param_memory =[]
eta = 0.1 #weight for parameters memory

def drop_momentum(m,p):
    r=p/1-p
    drop_m = m * torch.normal(mean=torch.ones_like(m), std=torch.ones_like(m)*r)
    
    return drop_m


class inner_SGD(optim.SGD):

    def __init__(self, *args, detach=False, **kwargs):
        self.detach = detach
        super(inner_SGD, self).__init__(*args, **kwargs)

    def step(self, closure=None, retain_graph=False, flag=False,i_m=None, task_lr=None, idx=None):
        global pg_memory
        global new_param_memory
        
        if not retain_graph:
            return super(inner_SGD, self).step(closure)

        loss = None
        if closure is not None:
            loss = closure()

        if flag is True:
            new_params=i_m
        else:
            new_params = []
            if idx ==0:
                new_param_memory = []
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                i=0
                if idx == 0:
                    pg_memory = []

                new_pg = []
                for p in group['params']:
                    i = 0
                    if p.grad is None:
                        new_params.append(p)
                        continue

                    d_p = p.grad if not self.detach else p.grad.detach()

                    if weight_decay != 0:
                        d_p = d_p.add(weight_decay, p)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = \
                                torch.zeros_like(p.data)
                            buf = buf.mul(momentum).add(d_p)
                        else:
                            buf = param_state['momentum_buffer'].to(p.device)
                            buf = buf.mul(momentum).add(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p + momentum * buf
                        else:
                            d_p = buf

                    #### GMD
                    p=0.1#dropout rate
                    dp = drop_momentum(dp, 0.1)

                    # the Mixed Gradient Preprocessing (MGP)
                    # task_lr is meta-leaned by ALR
                    # dp is the dropped momentum by GMD
                    if task_lr is None:
                        p = p - group['lr'] * d_p
                    else:
                        p = p - task_lr * d_p

                    p.retain_grad()

                    new_params.append(p)
                    new_pg.append(p)

                # memory regularization
                if idx == 0:
                    if i == 0:
                        pg_memory = []
                    pg_memory.append(new_pg)
                elif idx % W == 0:
                    new_pg = (1 - eta) * new_pg + eta * pg_memory[i]
                    pg_memory[i] = new_pg

                group['params'] = new_pg
                i=i+1

            if idx == 0:
                new_param_memory = new_params
            elif idx % W == 0:
                new_params = (1 - eta) * new_params + eta * new_param_memory
                new_param_memory = new_params

        return loss, new_params

class outer_SGD(optim.SGD):

    def __init__(self, *args, detach=False, **kwargs):
        self.detach = detach
        super(outer_SGD, self).__init__(*args, **kwargs)

    def step(self, closure=None, retain_graph=False, flag=True):
        if not retain_graph:
            return super(outer_SGD, self).step(closure)

        loss = None
        if closure is not None:
            loss = closure()

        new_params = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            new_pg = []
            for p in group['params']:
                if p.grad is None:
                    new_params.append(p)
                    continue

                d_p = p.grad if not self.detach else p.grad.detach()

                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                        buf = buf.mul(momentum).add(d_p)
                    else:
                        buf = param_state['momentum_buffer'].to(p.device)
                        buf = buf.mul(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                p = p - group['lr'] * d_p

                p.retain_grad()

                new_params.append(p)
                new_pg.append(p)
            group['params'] = new_pg
        return loss, new_params
