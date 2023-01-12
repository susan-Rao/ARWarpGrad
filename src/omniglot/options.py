"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We modify the code to provide ARWarpGrad.

"""
import argparse
def parse_set():
    parser = argparse.ArgumentParser("ARWarpGrad")

    # #for omniglot
    parser.add_argument('--root', type=str, default='data',
                        help="The number of classes to predict at any given draw ")

    # #for miniimagenet
    # parser.add_argument('--root', type=str, default='/data',
    #                     help="The number of classes to predict at any given draw")
    # parser.add_argument('--data_dir', type=str, default='/data/rsz/mini-imagenet',
    #                     help="The number of classes to predict at any given draw for 2224")

    parser.add_argument('--seed', type=int, default=8879,
                        help="The seed to use")
    parser.add_argument('--workers', type=int, default=0,
                        help="Data-loading parallelism")

    parser.add_argument('--num_pretrain', type=int, default=20,
                        help="Number of tasks to meta-train on")
    parser.add_argument('--classes', type=int, default=20,
                        help="Number of classes in a task")

    parser.add_argument('--meta_batch_size', type=int, default=5,
                        help="Tasks per meta-batch")
    parser.add_argument('--task_batch_size', type=int, default=100,
                        help="Samples per task-batch")
    # for n-way k-shot settings
    # parser.add_argument('--num_sample', type=int, default=5,
    #                     help="Samples per class")
    parser.add_argument('--num_query', type=int, default=5,
                        help="Samples per class for validation")

    parser.add_argument('--meta_train_steps', type=int, default=1000,
                        help="Number of steps in the outer (meta) loop")

    parser.add_argument('--task_train_steps', type=int, default=20,
                        help="Number of steps in the inner (task) loop")
    parser.add_argument('--task_val_steps', type=int, default=20,
                        help="Number of steps when training on validation tasks")

    parser.add_argument('--log_ival', type=int, default=1,
                        help="Interval between logging to stdout")
    parser.add_argument('--write_ival', type=int, default=1,
                        help="Interval between logging to file")
    parser.add_argument('--test_ival', type=int, default=20,
                        help="Interval between evaluating on validation set")

    parser.add_argument('--device', type=int, default=0,
                        help="Index for GPU device")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="Directory to write logs to")
    parser.add_argument('--suffix', type=str, default='Omniglot_multishot',
                        help="Name of experiment")
    parser.add_argument('--overwrite', type=bool, default=True,
                        help='Allow overwrite of existing log dir (same suffix)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate saved model')

    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size in conv layers')
    parser.add_argument('--padding', type=int, default=1,
                        help='Padding in conv layers')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of convolution layers in classifier')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Number of filters in each conv layer')
    parser.add_argument('--no_batch_norm', action='store_true',
                        help='Turn off batch normalization')

    parser.add_argument('--meta_model', type=str, default='ARWarpGrad')
    parser.add_argument('--inner_opt', type=str, default='SGD',
                        help='Optimizer in inner (task) loop: SGD or Adam')
    parser.add_argument('--outer_opt', type=str, default='SGD',
                        help='Optimizer in outer (meta) loop: SGD or Adam')

    # SGDM
    parser.add_argument('--inner_kwargs', nargs='+', default=['lr', '0.05', 'momentum', '0.1'],
                        help='Kwargs for inner optimizer')
    # SGD
    # parser.add_argument('--inner_kwargs', nargs='+', default=['lr', '0.05'],
    #                     help='Kwargs for inner optimizer')
    parser.add_argument('--outer_kwargs', nargs='+', default=['lr', '0.05'],
                        help='Kwargs for outer optimizer')
    parser.add_argument('--meta_kwargs', nargs='+', default=[],
                        help='Kwargs for meta learner')

    parser.add_argument('--warp_num_layers', type=int, default=1,
                        help='Number of conv layers in a block of warp layers.')
    parser.add_argument('--warp_num_filters', type=int, default=64,
                        help='Number of out filters in warp convolutions.')
    parser.add_argument('--warp_act_fun', type=str, default='None',
                        help='Warp-layer activation function.')
    parser.add_argument('--warp_residual', action='store_true',
                        help='Residual connection in warp-layer.')
    parser.add_argument('--warp_batch_norm', action='store_true',
                        help='Batch norm in warp-layer.')
    parser.add_argument('--warp_final_head', action='store_true',
                        help='Warp final linear layer.')
    return parser