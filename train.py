import time
import os
import torch
import torch.distributed as dist
from config import device


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):
    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()

    for idx, data in enumerate(train_loader):

        images = data[0]
        targets = data[1]
        images = images.to(device)
        outputs = model(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = criterion(outputs, targets)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()

        lr = []
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])

        # for each steps
        if idx % opts.vis_step == 0 or idx == len(train_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate DETR: {lr1:.7f} s \t'
                  'Learning rate backbone : {lr2:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch,
                          idx,
                          len(train_loader),
                          loss=loss,
                          lr1=lr[0],
                          lr2=lr[1],
                          time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss']))

    # save pth file
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))
