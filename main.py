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
# from losses.original_losses import HungarianLoss
from losses.matcher import HungarianMatcher

cudnn.benchmark = True


def main():
    opts = parse(sys.argv[1:])
    
    # 3. visdom
    # vis = visdom.Visdom(port=opts.port)

    # 4. data set
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_val = T.Compose([
        T.RandomResize([600], max_size=600),
        normalize,
    ])

    coco_dataset = COCO_Dataset(root=opts.data_root,
                                split='val',
                                download=True,
                                transforms=transforms_val,
                                visualization=False)
    # 5. data loader
    data_loader = torch.utils.data.DataLoader(coco_dataset,
                                              batch_size=opts.batch_size,
                                              collate_fn=coco_dataset.collate_fn,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    # 6. network
    model = DETR(num_classes=91, num_queries=100).to(device)

    # 7. criterion
    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=91, matcher=matcher).to(device)
    # 8. optimizer

    # 9. scheduler

    # 10. resume

    # 11. Train Start
    model.train()
    criterion.train()

    for i, data in enumerate(data_loader):

        images = data[0]
        targets = data[1]

        images = images.to(device)
        outputs = model(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = criterion(outputs, targets)
        print(loss)


if __name__ == "__main__":
    main()