import torch
import os
import sys
import visdom

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

    # 7. criterion

    # 8. optimizer

    # 9. scheduler

    # 10. resume

    # 11. Train Start

if __name__ == "__main__":
    main()