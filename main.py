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

cudnn.benchmark = True


def main():
    # 1. configuration
    opts = parse(sys.argv[1:])
    
    # 2. visdom
    vis = None
    if opts.data_root == "D:/data/coco":
        # for window
        vis = visdom.Visdom(port='8097')

    # 3. dataset
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_val = T.Compose([
        T.RandomResize([600], max_size=600),
        normalize,
    ])

    train_set = COCO_Dataset(root=opts.data_root,
                             split='train',
                             download=True,
                             transforms=transforms_val,
                             visualization=False)

    test_set = COCO_Dataset(root=opts.data_root,
                            split='val',
                            download=True,
                            transforms=transforms_val,
                            visualization=False)

    # 4. dataloader
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

    # 5. model (opts.num_classes = 91)
    model = DETR(num_classes=opts.num_classes, num_queries=100).to(device)

    # 6. criterion
    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=opts.num_classes, matcher=matcher).to(device)

    # 7. optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=opts.weight_decay)

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
        # test(epoch=epoch,
        #      vis=vis,
        #      test_loader=test_loader,
        #      model=model,
        #      criterion=criterion,
        #      opts=opts)


if __name__ == "__main__":
    main()