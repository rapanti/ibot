# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import itertools as itt
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import utils
import wids
from models.head import iBOTHead
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_args_parser():
    parser = argparse.ArgumentParser("iBOT", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "deit_tiny",
            "deit_small",
            "swin_tiny",
            "swin_small",
            "swin_base",
            "swin_large",
        ],
        help="Name of architecture to train.",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels of input square patches - default 16 (for 16x16 patches). Using smaller values leads to
        better performance but requires more memory. Applies only for ViTs (vit_tiny, vit_small and vit_base).
        If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--window_size",
        default=7,
        type=int,
        help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""",
    )
    parser.add_argument(
        "--out_dim",
        default=8192,
        type=int,
        help="""Dimensionality of output for [CLS] token.""",
    )
    parser.add_argument(
        "--patch_out_dim",
        default=8192,
        type=int,
        help="""Dimensionality of output for patch tokens.""",
    )
    parser.add_argument(
        "--shared_head",
        default=False,
        type=utils.bool_flag,
        help="""Wether to share the same head for [CLS] token output and patch tokens output. When set to false,
        patch_out_dim is ignored and enforced to be same with out_dim. (Default: False)""",
    )
    parser.add_argument(
        "--shared_head_teacher",
        default=True,
        type=utils.bool_flag,
        help="""See above. Only works for teacher model. (Defeault: True)""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--norm_in_head", default=None, help="Whether to use batch normalizations in projection head (Default: None)"
    )
    parser.add_argument(
        "--act_in_head", default="gelu", help="Whether to use batch normalizations in projection head (Default: gelu)"
    )
    parser.add_argument(
        "--use_masked_im_modeling",
        default=True,
        type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)",
    )
    parser.add_argument(
        "--pred_ratio",
        default=0.3,
        type=float,
        nargs="+",
        help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""",
    )
    parser.add_argument(
        "--pred_ratio_var",
        default=0,
        type=float,
        nargs="+",
        help="""Variance of partial prediction ratio.
        Length should be indentical to the length of pred_ratio. 0 for disabling. """,
    )
    parser.add_argument("--pred_shape", default="block", type=str, help="""Shape of partial prediction.""")
    parser.add_argument(
        "--pred_start_epoch",
        default=0,
        type=int,
        help="""Start epoch to perform masked image prediction.
        We typically set this to 50 for swin transformer. (Default: 0)""",
    )
    parser.add_argument(
        "--lambda1",
        default=1.0,
        type=float,
        help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""",
    )
    parser.add_argument(
        "--lambda2",
        default=1.0,
        type=float,
        help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_patch_temp",
        default=0.04,
        type=float,
        help="""See `--warmup_teacher_temp`""",
    )
    parser.add_argument(
        "--teacher_patch_temp",
        default=0.07,
        type=float,
        help=""""See `--teacher_temp`""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=30,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not to use half precision for training.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=128,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of linear warmup (highest LR used during training).
        The LR is linearly scaled with the batch size and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up."
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument("--load_from", default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument("--drop_path", type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_number",
        type=int,
        default=2,
        help="""Number of global views to generate. Default is to use two global crops.""",
    )
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.14, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=0,
        help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )

    parser.add_argument("--use_hvp", type=utils.bool_flag, default=False, help="Whether to use HVP.")
    parser.add_argument(
        "--hvp_step",
        type=int,
        default=1,
        help="Step to perform HVP. If set to 1, HVP is performed at each iteration.",
    )
    parser.add_argument(
        "--hvp_limit",
        type=int,
        default=0,
        help="Limit the number of combinations to check.",
    )
    parser.add_argument(
        "--global_crops_number_loader",
        type=int,
        default=None,
        help="Number of global crops for data loader.",
    )
    parser.add_argument(
        "--local_crops_number_loader",
        type=int,
        default=None,
        help="Number of local crops for data loader.",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument("--output_dir", default=".", type=str, help="Path to save logs and checkpoints.")
    parser.add_argument("--saveckp_freq", default=0, type=int, help="Save checkpoint every x epochs.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    return parser


def train_ibot(args):
    assert args.global_crops_number == 2, "Only 2 global crops are supported for now"
    assert (
        args.global_crops_number_loader is not None and args.local_crops_number_loader is not None
        if args.use_hvp
        else True
    ), "_crops_number_loader should be specified if using HVP"
    if not args.use_hvp:
        args.global_crops_number_loader = args.global_crops_number
        args.local_crops_number_loader = args.local_crops_number
    assert (
        args.local_crops_number_loader >= args.local_crops_number
    ), "local_crops_number_loader should be larger than local_crops_number"
    assert (
        args.global_crops_number_loader >= args.global_crops_number
    ), "global_crops_number_loader should be larger than global_crops_number"

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        args.global_crops_number_loader,
        args.local_crops_number_loader,
    )

    pred_size = args.patch_size * 8 if "swin" in args.arch else args.patch_size
    make_sample = SampleMaker(
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    trainset = wids.ShardListDataset(
        os.path.join(args.data_path, 'train', "imagenet_train.json"),
        cache_dir="/tmp",
        keep=True,
    )
    trainset.add_transform(make_sample)

    trainsampler = wids.DistributedChunkedSampler(
        trainset, chunksize=1000, shuffle=True, shufflefirst=True, seed=args.seed
    )

    data_loader = DataLoader(
        trainset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        sampler=trainsampler,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(trainset) // (args.batch_size_per_gpu * utils.get_world_size())
    args.steps_per_epoch = steps_per_epoch

    print(f"Data loaded: there are {len(trainset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and "swin" in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = (
            nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False)
            if "swin" in args.arch
            else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = (
        nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False)
        if "swin" in args.arch
        else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    )
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.global_crops_number_loader,
        args.local_crops_number,
        args.local_crops_number_loader,
        args.hvp_limit,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    # ============ preparing optimizer ... ============
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
        fp16_scaler = torch.cuda.amp.GradScaler()  # type: ignore

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        steps_per_epoch,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, steps_per_epoch)

    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        trainsampler.set_epoch(epoch)
        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(steps_per_epoch=args.steps_per_epoch, delimiter=" ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    for it, (images, masks, index) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # for it, (images, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = args.steps_per_epoch * epoch + it  # global training iteration

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

        index = index.cuda(non_blocking=True)
        tensor_list = [torch.zeros_like(index) for _ in range(args.world_size)]
        dist.all_gather(tensor_list, index)
        tensor_list = [t.cpu().detach().numpy() for t in tensor_list]
        if np.concatenate(tensor_list).size != np.unique(np.concatenate(tensor_list)).size:
            print("Error: intersection found")
            sys.exit()

        if args.use_hvp:
            if not it % args.hvp_step:
                images, masks = hard_view_selection(images, masks, student, teacher, ibot_loss, epoch, args)
            else:
                a = args.global_crops_number
                b = args.global_crops_number_loader
                c = args.local_crops_number
                images = images[:a] + images[b : b + c]
                masks = masks[:a]

        with torch.cuda.amp.autocast(fp16_scaler is not None):  # type: ignore
            # get global views
            teacher_output = teacher(images[: args.global_crops_number])
            student_output = student(images[: args.global_crops_number], mask=masks[: args.global_crops_number])

            # get local views
            student_local_cls = None
            if len(images) > args.global_crops_number:
                student.module.backbone.masked_im_modeling = False
                student_local_cls = student(images[args.global_crops_number :])[0]
                student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)  # type: ignore
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        ngcropsloader,
        nlcrops,
        nlcropsloader,
        limit,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.ngcropsloader = ngcropsloader
        self.nlcrops = nlcrops
        self.nlcropsloader = nlcropsloader
        self.ncrops = ngcrops + nlcrops
        self.ncropsloader = ngcropsloader + nlcropsloader
        self.limit = limit
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        a = itt.product(range(0, ngcropsloader, 2), range(1, ngcropsloader, 2))
        b = itt.combinations(range(ngcropsloader, ngcropsloader + nlcropsloader), nlcrops)
        ab = itt.product(a, b)
        self.combs = [list(itt.chain(*x)) for x in ab]
        assert limit < len(self.combs), "limit must be smaller than the number of combinations"

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2,
                )
            )
        )

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

    def hv_forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        bs = student_mask[0].size(0)
        device = student_mask[0].device

        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncropsloader)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcropsloader)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcropsloader)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcropsloader)

        score = torch.zeros(bs, device=device)
        selection = torch.zeros((2 + self.nlcrops, bs), dtype=torch.uint8, device=device)
        for idx in self.get_combinations(self.limit):
            _teacher_cls_c = [teacher_cls_c[x] for x in idx[:2]]
            _teacher_patch_c = [teacher_patch_c[x] for x in idx[:2]]
            _student_patch_c = [student_patch_c[x] for x in idx[:2]]
            _student_mask = [student_mask[x] for x in idx[:2]]
            _student_cls_c = [student_cls_c[x] for x in idx]

            total_loss1, n_loss_terms1 = 0, 0
            total_loss2, n_loss_terms2 = 0, 0
            for q in range(len(_teacher_cls_c)):
                for v in range(len(_student_cls_c)):
                    if v == q:
                        loss2 = torch.sum(-_teacher_patch_c[q] * F.log_softmax(_student_patch_c[v], dim=-1), dim=-1)
                        mask = _student_mask[v].flatten(-2, -1)
                        loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                        total_loss2 += loss2
                        n_loss_terms2 += 1
                    else:
                        loss1 = torch.sum(-_teacher_cls_c[q] * F.log_softmax(_student_cls_c[v], dim=-1), dim=-1)
                        total_loss1 += loss1
                        n_loss_terms1 += 1

            total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
            total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
            total_loss = total_loss1 + total_loss2

            score, indices = torch.stack((score, total_loss)).max(dim=0)  # type: ignore
            indices = indices.type(torch.bool)

            for n, ids in enumerate(idx):
                selection[n][indices] = ids

        return selection

    def get_combinations(self, limit):
        if limit == 0:
            for c in self.combs:
                yield c
        else:
            for idx in random.sample(range(len(self.combs)), limit):
                yield self.combs[idx]


class DataAugmentationiBOT(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crops_number_loader,
        local_crops_number_loader,
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )
        self.global_crops_number_loader = global_crops_number_loader
        self.local_crops_number_loader = local_crops_number_loader

    def __call__(self, image):
        crops = []
        for n in range(self.global_crops_number_loader):
            if n % 2 == 0:
                crops.append(self.global_transfo1(image))
            else:
                crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number_loader):
            crops.append(self.local_transfo(image))
        return crops


@torch.no_grad()
def hard_view_selection(images, masks, student, teacher, criterion, epoch, args):
    with torch.cuda.amp.autocast():  # type: ignore
        teacher_output = teacher(images[: args.global_crops_number_loader])
        student_output = student(
            images[: args.global_crops_number_loader], mask=masks[: args.global_crops_number_loader]
        )

        student_local_cls = None
        if len(images) > args.global_crops_number_loader:
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number_loader :])[0]
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

        selection = criterion.hv_forward(student_output, teacher_output, student_local_cls, masks, epoch)

    out_images = [torch.empty_like(images[0]) for _ in range(2)] + [
        torch.empty_like(images[-1]) for _ in range(args.local_crops_number)
    ]
    out_masks = [torch.empty_like(masks[0]) for _ in range(2)]

    # copy the selected images and masks
    # for global crops
    for n in range(args.global_crops_number):
        for m in range(args.global_crops_number_loader):
            out_images[n] = torch.where((selection[n] == m)[:, None, None, None], images[m], out_images[n])
            out_masks[n] = torch.where((selection[n] == m)[:, None, None], masks[m], out_masks[n])
    # for local crops
    for n in range(args.global_crops_number, len(out_images)):
        for m in range(args.global_crops_number_loader, len(images)):
            out_images[n] = torch.where((selection[n] == m)[:, None, None, None], images[m], out_images[n])

    # check that all images are selected correctly
    # for n, ids in enumerate(selection):
    #     for m, idx in enumerate(ids):
    #         assert torch.equal(out_images[n][m], images[idx][m])

    return out_images, out_masks


class SampleMaker(object):
    def __init__(
        self,
        *args,
        transform,
        patch_size,
        pred_ratio,
        pred_ratio_var,
        pred_aspect_ratio,
        pred_shape="block",
        pred_start_epoch=0,
        **kwargs,
    ):
        super(SampleMaker, self).__init__(*args, **kwargs)
        self.transform = transform
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = (
            pred_ratio_var[0] if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1 else pred_ratio_var
        )
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = (
                random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + self.pred_ratio_var)
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, sample, val=False):
        output = self.transform(sample[".jpg"])
        # print(sample['__index__'])
        index = sample["__index__"]

        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == "block":
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top : top + h, left : left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == "rand":
                mask = np.hstack(
                    [
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]
                ).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        # return output, masks
        return output, masks, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iBOT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
