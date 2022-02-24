import torch
import os
import sys
import visdom

from .models.detr import DETR

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    opts = parse(sys.argv[1:])
    
    # 3. visdom
    vis = visdom.Visdom(port=opts.port)

    train_set = None
    test_set = None

    # 4. data set
    # - dataset

    # 5. data loader

    # 6. network
    model = DETR(num_classes=81, num_queries=100)

    # 7. criterion

    # 8. optimizer

    # 9. scheduler

    # 10. resume

    # 11. Train Start

if __name__ == "__main__":
    main()