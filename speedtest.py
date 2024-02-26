# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models as torchvision_models

import models
import utils
from evaluation.unsupervised.unsup_cls import eval_pred
from loader import ImageFolderMask
from main_ibot import DataAugmentationiBOT, get_args_parser, hard_view_selection, iBOTLoss
from models.head import iBOTHead


def train_ibot(args):
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
    dataset = ImageFolderMask(
        args.data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)  # type: ignore
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

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
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))

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

    epoch_array = np.array([])
    data_array = np.array([])
    iter_array = np.array([])
    train_array = np.array([])
    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats, datas, trains, iters, epoch_time = train_one_epoch(
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
        # log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        # if utils.is_main_process():
        #     with (Path(args.output_dir) / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        epoch_array = np.append(epoch_array, epoch_time)
        data_array = np.append(data_array, datas)
        train_array = np.append(train_array, trains)
        iter_array = np.append(iter_array, iters)
    if utils.is_main_process():
        folder = f"hvp-step{args.hvp_step}" if args.use_hvp else "vanilla"
        Path(args.output_dir, folder).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(args.output_dir, folder, "epoch.npy"), epoch_array)
        np.save(os.path.join(args.output_dir, folder, "data.npy"), data_array)
        np.save(os.path.join(args.output_dir, folder, "train.npy"), train_array)
        np.save(os.path.join(args.output_dir, folder, "iter.npy"), iter_array)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print("Times:\tMean Std Median")
    print(f"Epoch:\t{epoch_array.mean()}\t{epoch_array.std()}\t{np.median(epoch_array)}")
    print(f"Iter:\t{iter_array.mean()}\t{iter_array.std()}\t{np.median(iter_array)}")
    print(f"Train:\t{train_array.mean()}\t{train_array.std()}\t{np.median(train_array)}")
    print(f"Data:\t{data_array.mean()}\t{data_array.std()}\t{np.median(data_array)}")


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
    metric_logger = MetricLogger(delimiter=" ")
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

    pred_labels, real_labels = [], []
    start = time.time()
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # global training iteration

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]

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
            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[args.global_crops_number :])[0] if len(images) > args.global_crops_number else None
            )
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)  # type: ignore
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

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
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    epoch_time = time.time() - start
    data_array, train_array, iter_array = metric_logger.get_time_arrays()
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict, data_array, train_array, iter_array, epoch_time


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(utils.SmoothedValue)
        self.delimiter = delimiter
        self.data_array = []
        self.train_array = []
        self.iter_array = []

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        iter_time = utils.SmoothedValue(fmt="{avg:.6f}")
        data_time = utils.SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        start_time = time.time()
        end = time.time()
        for obj in iterable:
            tmp = time.time()
            stop = tmp - end
            self.data_array.append(stop)
            data_time.update(stop)
            yield obj
            self.train_array.append(time.time() - tmp)
            stop = time.time() - end
            self.iter_array.append(stop)
            iter_time.update(stop)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.6f} s / it)".format(header, total_time_str, total_time / len(iterable)))

    def get_time_arrays(self):
        return self.data_array, self.train_array, self.iter_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iBOT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
    i = 0
    while i < 1000000:
        i += 1
