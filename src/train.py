import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
from .model.pspnet import get_model
from .model.transformer import MultiHeadAttentionOne
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
from .util import setup, cleanup, to_one_hot, batch_intersectionAndUnionGPU
from tqdm import tqdm
from .test import validate_transformer
from typing import Dict
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from typing import Tuple
from .util import load_cfg_from_cfg_file, merge_cfg_from_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running process rank {rank}.")
    setup(args, rank, world_size)
    print(args)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        np.random.seed(args.manual_seed + rank)
        torch.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)
        random.seed(args.manual_seed + rank)

    # ====== Model + Optimizer ======
    model = get_model(args).to(rank)

    if args.resume_weights:
        if os.path.isfile(args.resume_weights):
            print("=> loading weight '{}'".format(args.resume_weights))

            pre_weight = torch.load(args.resume_weights)['state_dict']

            pre_dict = model.state_dict()
            for index, (key1, key2) in enumerate(zip(pre_dict.keys(), pre_weight.keys())):
                if 'classifier' not in key1 and index < len(pre_dict.keys()):
                    if pre_dict[key1].shape == pre_weight[key2].shape:
                        pre_dict[key1] = pre_weight[key2]
                    else:
                        print('Pre-trained {} shape and model {} shape: {}, {}'.
                              format(key2, key1, pre_weight[key2].shape, pre_dict[key1].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)

            print("=> loaded weight '{}'".format(args.resume_weights))
        else:
            print("=> no weight found at '{}'".format(args.resume_weights))

        # Fix the backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # ====== Transformer ======
    trans_dim = args.bottleneck_dim

    transformer = MultiHeadAttentionOne(
        args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5
    ).to(rank)

    optimizer_transformer = get_optimizer(
        args,
        [dict(params=transformer.parameters(), lr=args.trans_lr * args.scale_lr)]
    )
    transformer = nn.SyncBatchNorm.convert_sync_batchnorm(transformer)
    transformer = DDP(transformer, device_ids=[rank])

    trans_save_dir = get_model_dir_trans(args)

    # ====== Data  ======
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    if args.debug:
        iter_per_epoch = 5
    else:
        iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

    log_iter = iter_per_epoch

    # ====== Training  ======
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        _, _ = do_epoch(
            args=args,
            train_loader=train_loader,
            iter_per_epoch=iter_per_epoch,
            model=model,
            transformer=transformer,
            optimizer_trans=optimizer_transformer,
            epoch=epoch,
            log_iter=log_iter,
        )

        val_Iou, val_loss = validate_transformer(
            args=args,
            val_loader=episodic_val_loader,
            model=model,
            transformer=transformer
        )

        if args.distributed:
            dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
            val_Iou /= world_size
            val_loss /= world_size

        if main_process(args):
            # Model selection
            if val_Iou.item() > max_val_mIoU:
                max_val_mIoU = val_Iou.item()

                os.makedirs(trans_save_dir, exist_ok=True)
                filename_transformer = os.path.join(trans_save_dir, f'best.pth')

                if args.save_models:
                    print('Saving checkpoint to: ' + filename_transformer)

                    torch.save(
                        {
                            'epoch': epoch,
                            'state_dict': transformer.state_dict(),
                            'optimizer': optimizer_transformer.state_dict()
                        },
                        filename_transformer
                    )

            print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

    if args.save_models and main_process(args):
        filename_transformer = os.path.join(trans_save_dir, 'final.pth')
        torch.save(
            {
                'epoch': args.epochs,
                'state_dict': transformer.state_dict(),
                'optimizer': optimizer_transformer.state_dict()
             },
            filename_transformer
        )

    cleanup()


def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def do_epoch(
        args: argparse.Namespace,
        train_loader: torch.utils.data.DataLoader,
        model: DDP,
        transformer: DDP,
        optimizer_trans: torch.optim.Optimizer,
        epoch: int,
        iter_per_epoch: int,
        log_iter: int
) -> Tuple[torch.tensor, torch.tensor]:

    loss_meter = AverageMeter()
    train_losses = torch.zeros(log_iter).to(dist.get_rank())
    train_Ious = torch.zeros(log_iter).to(dist.get_rank())

    iterable_train_loader = iter(train_loader)

    model.train()
    transformer.train()

    for i in range(iter_per_epoch):
        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()

        spprt_imgs = spprt_imgs.to(dist.get_rank(), non_blocking=True)
        s_label = s_label.to(dist.get_rank(), non_blocking=True)
        q_label = q_label.to(dist.get_rank(), non_blocking=True)
        qry_img = qry_img.to(dist.get_rank(), non_blocking=True)

        # ====== Phase 1: Train the binary classifier on support samples ======

        # Keep the batch size as 1.
        if spprt_imgs.shape[1] == 1:
            spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(
                2, 3, args.image_size, args.image_size
            )
            s_label_reshape = s_label.squeeze(0).expand(
                2, args.image_size, args.image_size
            ).long()
        else:
            spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
            s_label_reshape = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

        binary_cls = nn.Conv2d(
            args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
        ).cuda()

        optimizer = optim.SGD(binary_cls.parameters(), lr=args.cls_lr)

        # Dynamic class weights
        s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
        back_pix = np.where(s_label_arr == 0)
        target_pix = np.where(s_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            ignore_index=255
        )

        with torch.no_grad():
            f_s = model.module.extract_features(spprt_imgs_reshape)  # [n_task, c, h, w]

        for index in range(args.adapt_iter):
            output_support = binary_cls(f_s)
            output_support = F.interpolate(
                output_support, size=s_label.size()[2:],
                mode='bilinear', align_corners=True
            )
            s_loss = criterion(output_support, s_label_reshape)
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

        # ====== Phase 2: Train the transformer to update the classifier's weights ======
        # Inputs of the transformer: weights of classifier trained on support sets, features of the query sample.

        # Dynamic class weights used for query image only during training
        q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
        q_back_pix = np.where(q_label_arr == 0)
        q_target_pix = np.where(q_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(q_back_pix[0]) / len(q_target_pix[0])]).cuda(),
            ignore_index=255
        )

        model.eval()
        with torch.no_grad():
            f_q = model.module.extract_features(qry_img)  # [n_task, c, h, w]
            f_q = F.normalize(f_q, dim=1)

        # Weights of the classifier.
        weights_cls = binary_cls.weight.data

        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
            args.batch_size, 2, weights_cls.shape[1]
        )  # [n_task, 2, c]

        # Update the classifier's weights with transformer
        updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [n_task, 2, c]

        f_q_reshape = f_q.view(args.batch_size, args.bottleneck_dim, -1)  # [n_task, c, hw]

        pred_q = torch.matmul(updated_weights_cls, f_q_reshape).view(
            args.batch_size, 2, f_q.shape[-2], f_q.shape[-1]
        )  # # [n_task, 2, h, w]

        pred_q = F.interpolate(
            pred_q, size=q_label.shape[1:],
            mode='bilinear', align_corners=True
        )

        loss_q = criterion(pred_q, q_label.long())

        optimizer_trans.zero_grad()
        loss_q.backward()
        optimizer_trans.step()

        # Print loss and mIoU
        intersection, union, target = intersectionAndUnionGPU(
            pred_q.argmax(1), q_label, args.num_classes_tr, 255
        )

        if args.distributed:
            dist.all_reduce(loss_q)
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)

        mIoU = (intersection / (union + 1e-10)).mean()
        loss_meter.update(loss_q.item() / dist.get_world_size())

        if main_process(args):
            train_losses[i] = loss_meter.avg
            train_Ious[i] = mIoU

    print('Epoch {}: The mIoU {:.2f}, loss {:.2f}'.format(
        epoch + 1, train_Ious.mean(), train_losses.mean()
    ))

    return train_Ious, train_losses


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.epochs = 2
        args.n_runs = 2
        args.save_models = False

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)