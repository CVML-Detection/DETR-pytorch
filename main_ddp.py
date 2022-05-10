import torch
import os
import sys
import visdom

import torch.backends.cudnn as cudnn
import dataset.transforms as T

from models.detr import DETR
from dataset.coco_dataset import COCO_Dataset
from config import device, device_ids, parse
from losses.hungarian_loss import HungarianLoss
from losses.matcher import HungarianMatcher
from train import train
from test import test
from parallel import DataParallelModel, DataParallelCriterion

# for distributed_training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

cudnn.benchmark = True


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def main_worker(rank, world_size, opts, master_addr, master_port):
    # rank setting
    if opts.dist_mode == 'ddp':
        torch.cuda.set_device(opts.gpu_id)
        if opts.rank is not None:
            print("\nUse GPU: {} for training".format(opts.gpu_id))
            print("RANK: {}, World Size: {}".format(rank, world_size))
        # dist.init_process_group(backend='nccl',
        #                         init_method='tcp://127.0.0.1:3457',
        #                         world_size=world_size,
        #                         rank=opts.rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        print(f"{master_addr=} {master_port=}")
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        print("Use GPU(rank): {} for training | World Size : {}".format(torch.distributed.get_rank(), torch.distributed.get_world_size()))
        # dist.destroy_process_group()
        print('Process Group Loaded!')

    # 2. visdom
    if opts.visdom:
        vis = visdom.Visdom(port=opts.port)
    else:
        vis = None

    # 3. dataset
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # transforms_train #
    transforms_train = T.Compose([
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        T.RandomResize([800], max_size=800),
        normalize,
    ])

    # transforms_val #
    transforms_val = T.Compose([
        T.RandomResize([800], max_size=800),
        normalize,
    ])

    train_set = COCO_Dataset(root=opts.data_root,
                             split='train',
                             download=True,
                             transforms=transforms_train,
                             visualization=False)

    test_set = COCO_Dataset(root=opts.data_root,
                            split='val',
                            download=True,
                            transforms=transforms_val,
                            visualization=False)

    # 4. dataloader
    print('Data Loading...')
    # for DDP
    if opts.dist_mode == 'ddp':
        train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=int(opts.batch_size/world_size),
                                                collate_fn=train_set.collate_fn,
                                                shuffle=True,
                                                num_workers=int(opts.num_workers/world_size),
                                                pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=opts.batch_size,
                                                collate_fn=train_set.collate_fn,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=1,
                                            collate_fn=test_set.collate_fn,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True)
    print('Data Loaded!')


    # 5. model (opts.num_classes = 91)
    if opts.distributed:
        if opts.dist_mode == 'ddp':
            model = DETR(num_classes=opts.num_classes, num_queries=100).cuda(opts.gpu_id)
            model = DDP(module=model, device_ids=[opts.gpu_id], find_unused_parameters=True)
        elif opts.dist_mode == 'dp':
            model = DETR(num_classes=opts.num_classes, num_queries=100)
            model = torch.nn.DataParallel(model)
            model = model.cuda()
    else:
        model = DETR(num_classes=opts.num_classes, num_queries=100)
        model = model.cuda()


    # 6. criterion
    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=opts.num_classes, matcher=matcher)
    criterion.cuda()        # FIXME FIXME check


    # 7. optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": opts.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=opts.lr,
                                  weight_decay=opts.weight_decay)

    # 8. scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    # 9. resume
    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1),
                                map_location=torch.device('cuda:{}'.format(0)))
        model.load_state_dict(checkpoint['model_state_dict'])                          # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                  # load optimization state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                  # load scheduler state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:
        print('\nNo check point to resume.. train from scratch.\n')

    for epoch in range(opts.start_epoch, opts.epoch):

        # 10. train
        train(epoch=epoch,
              vis=vis,
              train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              opts=opts)

        # 11. test
        test(epoch=epoch,
             vis=vis,
             test_loader=test_loader,
             model=model,
             criterion=criterion,
             opts=opts,
             visualize=False)


def main():
    # 1. configuration
    opts = parse(sys.argv[1:])

    world_size = len(device_ids)
    master_addr = '127.0.0.1'
    master_port = find_free_port()

    if opts.dist_mode == 'ddp':
        mp.spawn(main_worker,
                args=(world_size, opts, master_addr, master_port),
                nprocs=world_size,
                join=True)
    else:
        main_worker(opts.gpu_id, world_size, opts, master_addr, master_port)


    

if __name__ == "__main__":
    main()