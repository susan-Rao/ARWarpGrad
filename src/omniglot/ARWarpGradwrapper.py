"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We modify the code to provide ARWarpGrad.

"""
import random
import torch
from abc import abstractmethod
from torch import nn
from torch.autograd import  Variable
import math

import sys
sys.path.append("..")

from ARWarp_utils.ARWarp_utils import updaters
from ARWarp_utils.ARWarp_utils import WarpGrad
from ARWarp_utils.ARWarp_utils import adaptive_optim as optim

from utils import Res, AggRes
from adaptive_lr import generate_lr_optimizer


K = 100
eps = 1e-6

###===###
# The following code is used for pre-processing the raw gradient
def preprocess_gradients(x):
    p = 10
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    x1 = x1.unsqueeze(1)
    x2 = x2.unsqueeze(1)

    return torch.cat((x1, x2), 1)


class ARBaseWrapper(object):
    def __init__(self, criterion, model, optimizer_cls, optimizer_kwargs):
        self.criterion = criterion
        self.model = model
        self.device = next(self.model.parameters()).device
        self.init_module = nn.Sequential(
                nn.Linear(18, 18),
                nn.ReLU(inplace=True),
                nn.Linear(18, 18),
                nn.Sigmoid()
            ).to(device=self.device)


        self.optimizer_cls = \
            optim.inner_SGD if optimizer_cls.lower() == 'sgd' else optim.Adam
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, tasks, meta_train=True):
        return self.run_tasks(tasks, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        NotImplementedError('Implement in meta-learner class wrapper.')

    def initialization_modulation(self, task_embeddings, params):
        v = self.init_module(task_embeddings)

        # vs = []
        # for i in range(v.size(0)):
        #     vs.append(v[i])

        updated_params = list(map(
            lambda current_params, v: ((v) * current_params['params'].to(device=self.device)),
            params,
            v))

        for i in range(len(params)):
            params[i]['params']=params[i]['params']+eps * updated_params[i]

        return params

    def run_tasks(self, tasks, meta_train):
        results = []
        for task in tasks:
            task.dataset.train()
            trainres = self.run_task(task, train=True, meta_train=meta_train)
            task.dataset.eval()
            valres = self.run_task(task, train=False, meta_train=False)

            # for n-way k-shot settings
            # trainres = self.run_task(task['train'], train=True, meta_train=meta_train)
            # valres = self.run_task(task['test'], train=False, meta_train=False)

            results.append((trainres, valres))
        ##
        results = AggRes(results)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return results

    def run_task(self, task, train, meta_train):
        optimizer = None
        if train:
            self.model.init_adaptation()
            self.model.train()
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train)

    def run_batches(self, batches, optimizer, train=False, meta_train=False, inner_lr=None):

        # ALR
        lr_generator=generate_lr_optimizer(model=self.model, PC=113236, HD=20,lr_base=inner_lr)

        res = Res()
        N = len(batches)

        for n, (input, target) in enumerate(batches):
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)


            global K #the total number of task-adaptation steps
            for k in range(0,K):
                # Evaluate model
                prediction = self.model(input)
                if target.size() == torch.Size([1]):
                    prediction = torch.unsqueeze(prediction, dim=0)
                loss = self.criterion(prediction, target)
                if not train:
                    continue
                final = (k == K)
                loss.backward()

                # initialization_modulation
                if k==0:
                    params = self.model.optimizer_parameter_groups()[0]
                    mean_grads = []
                    for tmp in params:
                        mean_grads.append(tmp['params'].grad.data.view(-1).mean())
                    task_embeddings = torch.stack(mean_grads)
                    # get the modulation parameters based on gradients, and update the parameters
                    updated_parameters = self.initialization_modulation(task_embeddings, params)

                    # update the model
                    optimizer.step(flag=True,i_m=updated_parameters)
                    # self.model.init_adaptation(updated_parameters)
                    # self.model.task_initialization(updated_parameters)

                    # clear the gradients
                    optimizer.zero_grad()
                    continue

                if meta_train:
                    self._partial_meta_update(loss, final)

                if inner_lr is None:
                    # without ALR
                    optimizer.step(idx=k)
                else:
                    # with ALR
                    params = self.model.optimizer_parameter_groups()[0]
                    grads = []
                    params_data = []
                    for tmp in params:
                        params_data.append(tmp['params'].data.view(-1))
                        grads.append(tmp['params'].grad.data.view(-1))

                    pre_grads = preprocess_gradients(torch.cat(grads))

                    inputs = Variable(torch.cat((pre_grads, torch.cat(params_data).unsqueeze(1)),1))
                    inner_lr=lr_generator(inputs,torch.cat(grads))

                    optimizer.step(task_lr=inner_lr,idx=k)

                optimizer.zero_grad()

                if final:
                    break

            res.log(loss=loss.item(), pred=prediction, target=target)

        ###
        res.aggregate()
        return res


class ARWarpGradWrapper(ARBaseWrapper):

    def __init__(self,
                 model,
                 optimizer_cls,
                 meta_optimizer_cls,
                 optimizer_kwargs,
                 meta_optimizer_kwargs,
                 meta_kwargs,
                 criterion):

        replay_buffer = WarpGrad.ReplayBuffer(
            inmem=meta_kwargs.pop('inmem', True),
            tmpdir=meta_kwargs.pop('tmpdir', None))

        optimizer_parameters = WarpGrad.OptimizerParameters(
            trainable=meta_kwargs.pop('learn_opt', False),
            default_lr=optimizer_kwargs['lr'],
            default_momentum=optimizer_kwargs['momentum']
            if 'momentum' in optimizer_kwargs else 0.)


        updater = updaters.DualUpdater(criterion, **meta_kwargs)

        model = WarpGrad.Warp(model=model,
                              adapt_modules=list(model.adapt_modules()),
                              warp_modules=list(model.warp_modules()),
                              updater=updater,
                              buffer=replay_buffer,
                              optimizer_parameters=optimizer_parameters)

        super(ARWarpGradWrapper, self).__init__(criterion,
                                              model,
                                              optimizer_cls,
                                              optimizer_kwargs)

        self.meta_optimizer_cls = optim.outer_SGD \
            if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        lra = meta_optimizer_kwargs.pop(
            'lr_adapt', meta_optimizer_kwargs['lr'])
        lri = meta_optimizer_kwargs.pop(
            'lr_init', meta_optimizer_kwargs['lr'])
        lrl = meta_optimizer_kwargs.pop(
            'lr_lr', meta_optimizer_kwargs['lr'])
        self.meta_optimizer = self.meta_optimizer_cls(
            [{'params': self.model.init_parameters(), 'lr': lri},
             {'params': self.model.warp_parameters(), 'lr': lra},
             {'params': self.model.optimizer_parameters(), 'lr': lrl}],
            **meta_optimizer_kwargs)

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):

        def step_fn():
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        self.model.backward(step_fn, **self.optimizer_kwargs)

    def run_task(self, task, train, meta_train):
        if meta_train and train:
            # Register new task in buffer.
            self.model.register_task(task)
            self.model.collect()
        else:
            # Make sure we're not collecting non-meta-train data
            self.model.no_collect()

        optimizer = None
        if train:
            # Initialize model adaptation
            self.model.init_adaptation()

            optimizer = self.optimizer_cls(
                self.model.optimizer_parameter_groups()[0],
                **self.optimizer_kwargs)

            if self.model.collecting and self.model.learn_optimizer:
                # Register optimiser to collect potential momentum buffers
                self.model.register_optimizer(optimizer)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train, inner_lr=self.model.optimizer_parameter_groups()[1])
