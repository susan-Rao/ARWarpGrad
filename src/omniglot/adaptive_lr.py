"""

Code for ARWarpGrad.

"""
import torch
from torch import nn
from torch.autograd import  Variable
from functools import reduce
from operator import mul

class generate_lr_optimizer(nn.Module):
    def __init__(self,model, PC, HD,lr_base):
        super(generate_lr_optimizer, self).__init__()
        ###===###
        self.RefM = model
        self.HD = HD
        self.PC = PC

        ###===###
        self.LSTM1 = nn.LSTMCell(HD, HD)
        self.PT1 = nn.Linear(PC, HD)
        self.PT2 = nn.Linear(5, 1)

        ###===###

        beta = []
        for tmp in lr_base:
            beta.append(tmp['params'].data.view(-1))
        self.beta_base = torch.cat(beta)

        self.W2 = nn.Linear(HD, PC)

        self.h1x = []
        self.c1x = []
        self.h2x = []
        self.c2x = []

        num_layers = 2

        for i in range(num_layers):
            self.h1x.append(Variable(torch.zeros(1, self.HD)))
            self.c1x.append(Variable(torch.zeros(1, self.HD)))
            self.h1x[i], self.c1x[i] = \
                self.h1x[i].cuda(), self.c1x[i].cuda()
            self.h2x.append(Variable(torch.zeros(1, self.HD)))
            self.c2x.append(Variable(torch.zeros(1, self.HD)))
            self.h2x[i], self.c2x[i] = \
                self.h2x[i].cuda(), self.c2x[i].cuda()

        self.cuda()

    def forward(self,pgrads,grad):
        ###===###
        # takes in all information
        x = pgrads
        y = grad.unsqueeze(1)
        z = torch.relu(self.beta_base).unsqueeze(1)
        pre_XI0 = torch.cat([x,y,z],dim=1).transpose(0,1)
        pre_XI1 = self.PT1(pre_XI0).transpose(0, 1)
        pre_XI2 = self.PT2(pre_XI1).squeeze(1)
        B_k = pre_XI2

        ###===###
        # loads the appropriate hidden states
        S_k = self.c1x[0]
        Q_k = self.h1x[0]

        B_k = B_k.unsqueeze(0)

        Q_k, S_k = self.LSTM1(B_k, (Q_k, S_k))
        self.h1x[0] = Q_k
        self.c1x[0] = S_k

        Z_k = Q_k.squeeze(0)


        self.i = torch.relu(self.beta_base + torch.tanh(self.W2(Z_k)))
        # self.i = self.set_flat_params(self.i)

        return self.i

    def set_flat_params(self, flat_params):

        offset = 0
        x = None

        for i, module in enumerate(self.RefM.children()):
            if isinstance(module, nn.Linear):
                weight_shape = module._parameters['weight'].size()
                bias_shape = module._parameters['bias'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)
                bias_flat_size = reduce(mul, bias_shape, 1)

                x['weight'] = flat_params[offset:offset + weight_flat_size].view(*weight_shape)
                x['bias'] = flat_params[offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

                offset += weight_flat_size + bias_flat_size

            if isinstance(module, nn.Conv2d):
                weight_shape = module._parameters['weight'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)

                x['weight'] = flat_params[offset:offset + weight_flat_size].view(*weight_shape)

                offset += weight_flat_size

        return x