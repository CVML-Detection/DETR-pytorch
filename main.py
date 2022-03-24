import torch
import os
import sys
import visdom

import torch.backends.cudnn as cudnn
import dataset.transforms as T

from models.detr import DETR
from dataset.coco_dataset import COCO_Dataset
from config import device, device_ids, parse

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
    model = DETR(num_classes=opts.num_classes, num_queries=100).to(device)

    # 7. criterion

    # 8. optimizer

    # 9. scheduler

    # 10. resume

    # 11. Train Start

    for i, data in enumerate(data_loader):
        images = data[0]
        targets = data[1]

        images = images.to(device)
        out_feat = model(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

if __name__ == "__main__":
    main()