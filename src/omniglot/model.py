"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We modify the code to provide ARWarpGrad.

"""

import torch.nn as nn
from ARWarpGradwrapper import ARWarpGradWrapper

Activation_functions = {
    'none': None,
    'leakyrelu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}

NUM_CLASSES = 50


def get_model(args, criterion):
    """Construct model from main args"""
    kwargs = dict(num_classes=args.classes,
                  num_layers=args.num_layers,
                  kernel_size=args.kernel_size,
                  num_filters=args.num_filters,
                  imsize=args.imsize,
                  padding=args.padding,
                  batch_norm=args.batch_norm,
                  multi_head=args.multi_head)
    model = WarpedOmniConv(warp_num_layers=args.warp_num_layers,
                               warp_num_filters=args.warp_num_filters,
                               warp_residual_connection=args.warp_residual,
                               warp_act_fun=args.warp_act_fun,
                               warp_batch_norm=args.warp_batch_norm,
                               warp_final_head=args.warp_final_head,
                               **kwargs)

    if args.cuda:
        model = model.cuda()

    return ARWarpGradWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            args.meta_kwargs,
            criterion)


###############################################################################


class UnSqueeze(nn.Module):

    """Create channel dim if necessary."""

    def __init__(self):
        super(UnSqueeze, self).__init__()

    def forward(self, input):
        if input.dim() == 4:
            return input
        return input.unsqueeze(1)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input):
        if input.size(0) != 0:
            return input.squeeze()
        input = input.squeeze()
        return input.view(1, *input.size())


class Linear(nn.Module):
    def __init__(self, multi_head, num_features_in,
                 num_features_out, **kwargs):
        super(Linear, self).__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

        self.multi_head = multi_head

        def _linear_factory():
            return nn.Linear(num_features_in, num_features_out, **kwargs)

        if self.multi_head:
            self.linear = nn.ModuleList([_linear_factory()] * NUM_CLASSES)
        else:
            self.linear = _linear_factory()

    def forward(self, x, idx=None):
        if self.multi_head:
            assert idx is not None, "Pass head idx in multi-headed mode."
            return self.linear[idx](x)
        return self.linear(x)

    def reset_parameters(self):
        if self.multi_head:
            for lin in self.linear:
                lin.reset_parameters()
        else:
            self.linear.reset_parameters()



class WarpLayer(nn.Module):
    def __init__(self, num_features_in, num_features_out,
                 kernel_size, padding, residual_connection,
                 batch_norm, act_fun):
        super(WarpLayer, self).__init__()
        self.residual_connection = residual_connection
        self.bn_in = None
        self.bn_out = None
        if batch_norm:
            self.bn_in = nn.BatchNorm2d(num_features_in)
            if self.residual_connection:
                self.bn_out = nn.BatchNorm2d(num_features_out)

        self.conv = nn.Conv2d(num_features_in,
                              num_features_out,
                              kernel_size,
                              padding=padding)

        self.act_fun = act_fun if act_fun is None else act_fun()

        if residual_connection and num_features_in != num_features_out:
            self.scale = nn.Conv2d(num_features_in, num_features_out, 1)
        else:
            self.scale = None

    def forward(self, x):
        h = x

        if self.bn_in is not None:
            h = self.bn_in(h)

        h = self.conv(h)

        if self.act_fun is not None:
            h = self.act_fun(h)

        if self.residual_connection:
            if self.scale is not None:
                x = self.scale(x)

            h = x + h

        if self.bn_out is not None:
            h = self.bn_out(h)

        return h


class WarpedOmniConv(nn.Module):
    def __init__(self,
                 num_classes,
                 num_layers=4,
                 kernel_size=3,
                 num_filters=64,
                 imsize=(28, 28),
                 padding=True,
                 batch_norm=True,
                 multi_head=False,
                 warp_num_layers=1,
                 warp_num_filters=64,
                 warp_residual_connection=False,
                 warp_act_fun=None,
                 warp_batch_norm=True,
                 warp_final_head=False):
        super(WarpedOmniConv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.batch_norm = batch_norm
        self.multi_head = multi_head
        self.warp_num_layers = warp_num_layers
        self.warp_num_filters = warp_num_filters
        self.warp_residual_connection = warp_residual_connection
        self.warp_act_fun = Activation_functions[warp_act_fun.lower()]
        self.warp_batch_norm = warp_batch_norm
        self.warp_final_head = warp_final_head
        self._conv_counter = 0
        self._warp_counter = 0

        def conv_block(nin):
            # Task adaptable conv block, same as OmniConv
            _block = [nn.Conv2d(nin,
                                num_filters,
                                kernel_size,
                                padding=padding),
                      nn.MaxPool2d(2)]
            if batch_norm:
                _block.append(nn.BatchNorm2d(num_filters))
            _block.append(nn.ReLU())
            return nn.Sequential(*_block)

        def warp_layer(nin, nout):
            # We use same kernel_size and padding as OmniConv for simplicity
            return WarpLayer(nin, nout, kernel_size, padding,
                             self.warp_residual_connection,
                             self.warp_batch_norm,
                             self.warp_act_fun)

        def block(nin):
            # Task-adaptable layer
            self._conv_counter += 1
            setattr(self, 'conv{}'.format(self._conv_counter), conv_block(nin))

            # Warp-layers
            nin = num_filters
            for _ in range(self.warp_num_layers):
                self._warp_counter = \
                    self._warp_counter % self.warp_num_layers + 1

                if self._warp_counter == self.warp_num_layers:
                    nout = num_filters
                else:
                    nout = self.warp_num_filters

                setattr(self, 'warp{}{}'.format(self._conv_counter,
                                                self._warp_counter),
                        warp_layer(nin, nout))

                nin = nout

        block(1)
        for _ in range(self.num_layers-1):
            block(num_filters)

        if self.warp_final_head:
            self.head = Linear(self.multi_head, num_filters, num_filters)
            self.warp_head = nn.Linear(num_filters, num_classes)
        else:
            self.head = Linear(self.multi_head, num_filters, num_classes)

        self.squeeze = Squeeze()

    def forward(self, x, idx=None):
        for i in range(1, self._conv_counter+1):
            # Task-adaptable layer
            x = getattr(self, 'conv{}'.format(i))(x)
            # Warp-layer(s)
            for j in range(1, self._warp_counter+1):
                x = getattr(self, 'warp{}{}'.format(i, j))(x)

        x = self.squeeze(x)
        x = self.head(x, idx)

        if self.warp_final_head:
            return self.warp_head(x)
        return x

    def adapt_modules(self):
        for i in range(1, self.num_layers+1):
            conv = getattr(self, 'conv{}'.format(i))
            yield conv
        yield self.head

    def warp_modules(self):
        for i in range(1, self.num_layers+1):
            for j in range(1, self.warp_num_layers+1):
                warp = getattr(self, 'warp{}{}'.format(i, j))
                yield warp

        if self.warp_final_head:
            yield self.warp_head

    def init_adaptation(self):
        self.head.reset_parameters()

        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()
