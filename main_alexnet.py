import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
from custom_data import CustomImageFolder

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('alexnet', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='alexnet', type=str, help='Model backbone architecture.')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained weights')

    # Training/optimization parameters
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer.""")

    # Misc
    parser.add_argument('--data_path', default='/om2/user/amarvi/dino/data/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--meta_file', type=str, help="custom dataset")
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============= preparing data =============
    IMAGE_RESIZE=256
    IMAGE_SIZE=224
    GRAYSCALE_PROBABILITY=0.2
    resize_transform      = transforms.Resize(IMAGE_RESIZE)
    random_crop_transform = transforms.RandomCrop(IMAGE_SIZE)
    center_crop_transform = transforms.CenterCrop(IMAGE_SIZE)
    grayscale_transform   = transforms.RandomGrayscale(p=GRAYSCALE_PROBABILITY)
    normalize             = transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)

    transform = transforms.Compose([resize_transform, 
                    random_crop_transform,
                    grayscale_transform,
                    transforms.ToTensor(),
                    normalize,
                    ])
    if args.meta_file:
        dataset = CustomImageFolder(args.data_path, args.meta_file, transform=transform)
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building network ================
    if args.arch in torchvision_models.__dict__.keys():
        # TODO: get num_classes from dataloader. 
        backbone = torchvision_models.__dict__[args.arch](pretrained=args.pretrained)
        # embed_dim = backbone.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    print('backbone:', backbone)
    print('data_loader:', len(dataset), data_loader)

    # ============ preparing loss & optimizer ==========
    loss = nn.CrossEntropyLoss().cuda()
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ====================
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")


    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        backbone=backbone,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss=loss,
    )
    start_epoch = to_restore["epoch"]


    start_time = time.time()
    print("Starting training !")

    for epoch in range(start_epoch, args.epoch):
        data_loader.sampler.set_epoch(epoch)

        # ================= train ===================
        train_stats = train_one_epoch()

        # ================ writing logs =============
        save_dict = {
            'backbone': backbone.state_dict(),
            'optimizaer': optimizer.state_dict(),
            'epoch': epoch+1,
            'args': args,
            'loss': loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('alexnet', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
