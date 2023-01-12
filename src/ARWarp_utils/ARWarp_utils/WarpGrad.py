"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We modify the code to provide ARWarpGrad.

"""
from collections import OrderedDict

import os
import uuid
import tempfile
import torch

from .utils import (copy, copy_opt, clone_state, load, clear,
                    unfreeze, freeze, zero_grad, get_groups)


class ReplayBuffer:
    def __init__(self, inmem=True, tmpdir=None):

        self.inmem = inmem
        self._data_buffer = {}
        self._state_buffer = {}
        self._optimizer_buffer = {}
        self._idx = {}
        if not inmem and tmpdir is None:
            tmpdir = tempfile.mkdtemp('_WGDTMP')
        self.tmpdir = tmpdir

    def clear(self):
        """Clear buffer."""
        self._data_buffer.clear()
        self._idx.clear()
        if self.inmem:
            self._state_buffer.clear()
            self._optimizer_buffer.clear()
        else:
            clear(self.tmpdir)

    def init(self, slot, data):
        if slot in self._idx:
            raise ValueError('slot {} already in buffer'.format(slot))
        self._idx[slot] = 0
        self._data_buffer[slot] = data

    def update(self, slot, state, buffer=None):
        assert slot in self._idx, 'slot not in buffer. Call init_slot first'
        self._idx[slot] += 1

        if self.inmem:
            if slot not in self._state_buffer:
                assert self._idx[slot] == 1
                self._state_buffer[slot] = []
                if buffer is not None:
                    self._optimizer_buffer[slot] = []

            self._state_buffer[slot].append(clone_state(state, device='cpu'))
            if buffer is not None:
                self._optimizer_buffer[slot].append(copy_opt(buffer))
            return

        if buffer is not None:
            raise NotImplementedError(
                "Putting optimizer parameters on disk not implemented.")

        fname = '{}_{}.{}'.format(slot, self._idx[slot], '.tar')
        fpath = os.path.join(self.tmpdir, fname)
        torch.save(state, fpath)

    @property
    def dataset(self):
        """Current replay buffer."""
        if self.inmem:
            if self._optimizer_buffer:
                return (self._data_buffer, self._state_buffer,
                        self._optimizer_buffer)
            return self._data_buffer, self._state_buffer

        param_cache = load(self.tmpdir)
        return self._data_buffer, param_cache


class OptimizerParameters:
    def __init__(self, trainable, default_lr, default_momentum):
        self._opt = None
        self._trainable = trainable
        self._param_names = []

        self._lr = []
        self._momentum = []

        self.default_lr = default_lr
        self.default_momentum = default_momentum

    def init(self, named_parameters):
        self._lr = []
        self._momentum = []
        self._param_names = []

        for n, p in named_parameters:
            self._param_names.append(n)
            pl = torch.tensor(self.default_lr,
                              device=p.device,
                              requires_grad=self.trainable)
            pm = torch.tensor(self.default_momentum,
                              device=p.device,
                              requires_grad=self.trainable)
            self._lr.append(pl)
            self._momentum.append(pm)

    @property
    def lr(self):
        """Learning rates."""
        for l in self._lr:
            if l.item() < 0:
                l.data.fill_(1e-6)
            yield l

    @property
    def momentum(self):
        """Momentum rates."""
        for m in self._momentum:
            if m.item() < 0:
                m.data.fill_(1e-6)
            yield m

    @property
    def trainable(self):
        """Trainable parameters flag."""
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        self._trainable = trainable

        if self._trainable and self.default_momentum == 0:
            self.default_momentum = 1e-4
            for m in self._momentum:
                m.data.fill_(self.default_momentum)

        for p in self.parameters():
            p.requires_grad = self._trainable

    def parameters(self):
        for p in self._lr + self._momentum:
            yield p

    def named_parameters(self):
        """Optimizer parameters."""
        for n, p in zip(self._param_names, self._lr):
            n += '.lr'
            yield n, p

        for n, p in zip(self._param_names, self._momentum):
            n += '.mom'
            yield n, p

    def groups(self, parameters,tensor):
        """Parameter groups."""
        return get_groups(parameters, self,tensor=tensor)


class ARParameters:
    def __init__(self, model, adapt_modules, warp_modules,
                 optimizer_parameters):
        self.model = model
        self.adapt_modules = adapt_modules
        self.warp_modules = warp_modules

        self._optimizer = None
        self._learn_optimizer = optimizer_parameters.trainable
        self._optimizer_parameters = optimizer_parameters
        self._optimizer_parameters.init(self.named_adapt_parameters())

        self._init_state = clone_state(self.adapt_state())
        self._init_parameters = [(n, p) for n, p in self._init_state.items()
                                 if p.requires_grad]

        self._inner_lr = [(n, torch.ones_like(p) * 1e-3) for n, p in self._init_state.items() if p.requires_grad]

    def task_initialization(self, new_parameters):
        copy(self._init_parameters, new_parameters)

    def set_parameters(self, new_parameters):
        copy(self.adapt_parameters(), new_parameters)

    def set_state(self, new_state):
        copy(self.adapt_state(), new_state)

    def init_state(self):
        return self._init_state

    def adapt_state(self):
        """Return state_dict for adapt modules."""
        model_state = self.model.state_dict(keep_vars=True)
        adapt_tensors = [id(t) for m in self.adapt_modules
                         for t in m.state_dict(keep_vars=True).values()]
        return OrderedDict((n, t) for n, t in model_state.items()
                           if id(t) in adapt_tensors)

    def adapt_parameters(self):
        """Adapt parameters."""
        for m in self.adapt_modules:
            for p in m.parameters():
                yield p

    def named_adapt_parameters(self):
        """Named adapt parameters."""
        adapt_ids = list(map(id, self.adapt_parameters()))
        for n, p in self.model.named_parameters():
            if id(p) in adapt_ids:
                yield n, p

    def parameters(self):
        """All parameters."""
        return self.model.parameters()

    def optimizer_buffer(self):
        buffer = None
        if self._optimizer is not None:
            # opt.state is not ordered in pytorch v1
            buffer = []
            param_names = [n for n, _ in self.adapt_state()]
            for n in param_names:
                # check since opt.state is a dict factory
                if n in self._optimizer.state:
                    buffer.append(self._optimizer.state[n])
        return buffer

    def optimizer_parameters(self):
        return self._optimizer_parameters.parameters()

    def named_optimizer_parameters(self):
        return self._optimizer_parameters.named_parameters()

    def init_parameters(self):
        for _, p in self.named_init_parameters():
            yield p

    def named_init_parameters(self, suffix='.init'):
        for n, p in self._init_parameters:
            if suffix is not None:
                n += suffix
            yield n, p


    def inner_lr(self):
        for _, p in self.named_inner_lr():
            yield p

    def named_inner_lr(self, suffix='.initlr'):
        for n, p in self._inner_lr:
            if suffix is not None:
                n += suffix
            yield n, p

    def warp_parameters(self):
        for m in self.warp_modules:
            for p in m.parameters():
                yield p

    def named_warp_parameters(self, suffix=None):
        meta_param_ids = list(map(id, self.warp_parameters()))
        for n, p in self.model.named_parameters():
            if id(p) in meta_param_ids:
                if suffix is not None:
                    n += suffix
                yield n, p

    def meta_parameters(self,
                        include_warp=True,
                        include_init=True,
                        include_innerlr=True,
                        include_optimizer=True):
        if self.learn_optimizer and include_optimizer:
            for p in self.optimizer_parameters():
                yield p

        if include_init:
            for p in self.init_parameters():
                yield p

        if include_innerlr:
            for p in self.inner_lr():
                yield p

        if include_warp:
            for p in self.warp_parameters():
                yield p

    def named_meta_parameters(self,
                              include_warp=True,
                              include_init=True,
                              include_innerlr=True,
                              include_opt=True):
        if self.learn_optimizer and include_opt:
            for n, p in self.named_optimizer_parameters():
                yield n, p

        if include_init:
            for n, p in self.named_init_parameters():
                yield n, p

        if include_innerlr:
            for n, p in self.named_inner_lr():
                yield n, p

        if include_warp:
            for n, p in self.named_warp_parameters():
                yield n, p

    def optimizer_parameter_groups(self, tensor=False):
        return self._optimizer_parameters.groups(self.adapt_parameters(), tensor),\
                self._optimizer_parameters.groups(self.inner_lr(), tensor)

    def register_optimizer(self, optimizer):
        self._optimizer = optimizer

    def unregister_optimizer(self):
        self._optimizer = None

    @property
    def learn_optimizer(self):
        return self._optimizer_parameters.trainable

    @learn_optimizer.setter
    def learn_optimizer(self, learn_optimizer):
        self._optimizer_parameters.trainable = learn_optimizer


class Warp(ARParameters):
    def __init__(self, model, adapt_modules, warp_modules,
                 updater, buffer, optimizer_parameters):
        super(Warp, self).__init__(model,
                                   adapt_modules,
                                   warp_modules,
                                   optimizer_parameters)
        self.updater = updater
        self._task = None
        self._collect = True
        self.buffer = buffer
        self.zero_meta_grads()
        self.zero_task_grads()

    def __call__(self, *inputs, cache_parameters=None):
        if cache_parameters is None:
            cache_parameters = self._collect
        if cache_parameters:
            self._dump()
        return self.model(*inputs)

    def register_task(self, data):
        self._task = uuid.uuid4().hex
        self.buffer.init(self._task, data)

    def init_adaptation(self, reset_adapt_parameters=None):
        self.model.init_adaptation()

        if reset_adapt_parameters is None:
            # Will be 0 if no meta-objective for initialization is specified
            reset_adapt_parameters = self.updater.init_objective

        if reset_adapt_parameters:
            copy(self.adapt_state(), self.init_state())

        freeze(self.meta_parameters())
        unfreeze(self.adapt_parameters())

        self.model.train()

    def clear(self):
        self.buffer.clear()

    def collect(self):
        self._collect = True

    def no_collect(self):
        self._collect = False

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def zero_meta_grads(self):
        zero_grad(list(self.meta_parameters()))

    def zero_task_grads(self):
        zero_grad(list(self.adapt_parameters()))

    def backward(self, *args, retain_trajectories=False,
                 retain_optimizer=False, **kwargs):
        collecting = self.collecting
        if collecting:
            self.no_collect()

        self.updater.backward(self, *args, **kwargs)

        if not retain_trajectories:
            self.clear()

        if not retain_optimizer:
            self.unregister_optimizer()

        if collecting:
            self.collect()

    def _dump(self):
        self.buffer.update(self._task,
                           self.adapt_state(),
                           self.optimizer_buffer())

    @property
    def collecting(self):
        return self._collect

    @property
    def dataset(self):
        return self.buffer.dataset
