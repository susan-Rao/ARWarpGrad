"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

We modify the code to provide ARWarpGrad.

"""
import time
import os
from os.path import join

from options import parse_set

from data_multishot import DataLoader

import torch
from torch import nn
from torchvision import transforms

from model import get_model
from utils import build_kwargs, log_status, write_train_res, plot_train_res, write_val_res, unlink, consolidate



# for the n-way k-shot setting
# from data_loader import split_omniglot_characters
# from data_loader import load_imagenet_images
# from data_loader import OmniglotTask
# from data_loader import ImageNetTask
# from data_loader import fetch_dataloaders


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define a training image loader for ImageNet (miniImageNet, tieredImageNet)
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

parser = parse_set()

args = parser.parse_args()

args.imsize = (28, 28)
args.cuda = True
args.batch_norm = not args.no_batch_norm
# for k-shot
# if args.num_sample ==5:
#     args.batch_norm = True
# else:
#     args.batch_norm = False


args.inner_kwargs = build_kwargs(args.inner_kwargs)
args.outer_kwargs = build_kwargs(args.outer_kwargs)
args.meta_kwargs = build_kwargs(args.meta_kwargs)
args.multi_head = args.meta_model.lower() == 'ft'


def pp(*inputs, **kwargs):
    if args.log_ival > 0:
        print(*inputs, **kwargs)


if args.cuda and not torch.cuda.is_available:
    raise ValueError("Cuda is not available. Run with --no_cuda")


torch.manual_seed(args.seed)


def main():
    data = DataLoader(
        root=args.root,
        num_pretrain_alphabets=args.num_pretrain,
        num_classes=args.classes,
        num_sample=args.task_batch_size,
        num_query=args.num_query,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=True,
    )

    # for n-way k-shot settings
    # mini-imagent
    # data = DataContainer(
    #     root=args.root,
    #     num_pretrain_alphabets=args.num_pretrain,
    #     num_classes=args.classes,
    #     num_sample=args.num_sample,
    #     num_query=args.num_query,
    #     transform=train_transformer_ImageNet,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )

    # if 'Omniglot' in args.data_dir:
    #     (meta_train_classes, meta_val_classes,
    #      meta_test_classes) = split_omniglot_characters(
    #          args.data_dir, args.SEED)
    #     task_type = OmniglotTask
    # elif ('mini-imagenet' in args.data_dir or
    #       'tieredImageNet' in args.data_dir):
    #     (meta_train_classes, meta_val_classes,
    #      meta_test_classes) = load_imagenet_images(args.data_dir)
    #     task_type = ImageNetTask
    # else:
    #     raise ValueError("I don't know your dataset")

    ###########################################################################

    log_dir = os.path.join(args.log_dir, args.meta_model, args.suffix)

    def evaluate(model: object, case: object, step: object) -> object:
        if args.write_ival > 0:
            torch.save(model, join(log_dir, 'model.pth.tar'))
        # iterator = []

        # for multi-shot
        if case == 'test':
            iterator = data.test
        else:
            iterator = data.val

        iterator = iterator(args.task_batch_size,
                            args.task_val_steps,
                            args.multi_head)

        ### for n-way k-shot
        # if case == 'test':
        #     for n_task in range(HOLD_OUT):
        #         task = task_type(meta_test_classes, args.classes,
        #                          args.num_sample, args.num_query)
        #         dataloaders = fetch_dataloaders(['train','test'], task)
        #         iterator.append(dataloaders)
        # else:
        #     for n_task in range(HOLD_OUT):
        #         task = task_type(meta_train_classes, args.classes,
        #                          args.num_sample, args.num_query)
        #         dataloaders = fetch_dataloaders(['train','test'], task)
        #         iterator.append(dataloaders)

        pp('Evaluating on {} tasks'.format(case))

        results = []
        for i, task in enumerate(iterator):
            if args.write_ival > 0:
                task_model = torch.load(join(log_dir, 'model.pth.tar'))
            else:
                task_model = model

            t = time.time()
            task_results = task_model([task], meta_train=False)
            t = time.time() - t

            results.append(task_results)

            if args.log_ival > 0:
                log_status(task_results, 'task={}'.format(i), t)

        if args.log_ival > 0:
            log_status(consolidate(results), 'Average', t)

        if args.write_ival > 0:
            write_val_res(results, step, case, log_dir)

        pp('-Done-')

    ###########################################################################

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        model = torch.load(join(log_dir, 'model.pth.tar'))
        evaluate(model, 'test', 0)
        return

    model = get_model(args, criterion)

    ###########################################################################
    if args.write_ival > 0:
        if os.path.exists(log_dir):
            assert args.overwrite, \
                "Path exists ({}). Use --overwrite or change suffix".format(
                    log_dir)
            unlink(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        with open(join(log_dir, 'args.log'), 'w') as f:
            f.write("%r" % args)

        with open(join(log_dir, 'args.log'), 'w') as f:
            f.write("%r" % args)

    pp('Initiating meta-training')
    train_step = 0
    try:
        evaluate(model, 'val', train_step)
        t = time.time()
        while True:
            # for multi-shot
            task_batch = data.train(args.meta_batch_size,
                                    args.task_batch_size,
                                    args.task_train_steps,
                                    args.multi_head)

            # for n-way k-shot
            # task_batch=[]
            # for n_task in range(args.meta_batch_size):
            #     task = task_type(meta_train_classes, args.classes,
            #                      args.num_sample, args.num_query)
            #     dataloaders = fetch_dataloaders(['train','test'], task)
            #     task_batch.append(dataloaders)

            results = model(task_batch, meta_train=True)

            train_step += 1
            if train_step % args.write_ival == 0:
                write_train_res(results, train_step, log_dir)
                # for visualization
                # plot_train_res(results, train_step, log_dir)

            if train_step % args.test_ival == 0:
                evaluate(model, 'val', train_step)
                pp("Resuming training")

            if args.log_ival > 0 and train_step % args.log_ival == 0:
                t = (time.time() - t) / args.log_ival
                log_status(results, 'step={}'.format(train_step), t)
                t = time.time()

            if results.train_loss != results.train_loss:
                break

            if train_step == args.meta_train_steps:
                break

    except KeyboardInterrupt:
        pp('Meta-training stopped.')
    else:
        pp('Meta-training complete.')


if __name__ == '__main__':
    if args.cuda:
        with torch.cuda.device(args.device):
            main()
    else:
        pp('Warning: No GPU is selected!')
        main()
