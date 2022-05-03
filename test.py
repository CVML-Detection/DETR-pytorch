import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from config import device
from utils import box_cxcywh_to_xyxy, detect
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import coco_label_map as label_map
from utils import coco_color_array
from evaluator import Evaluator
from utils import coco_label_idx_91 as label_idx_91

coco_color_array = np.random.randint(256, size=(91, 3)) / 255  # In plt, rgb color space's range from 0 to 1
label_array = list(label_map.keys())  # dict
label_dict = label_idx_91


def visualize_results(images, results):
    '''
    :param images:
    :param results: [{'scores': s, 'labels': l, 'boxes': b}]
    :param label_map:
    :param color_array:
    :return:
    '''
    color_array = coco_color_array

    # 0. permute
    images = images.cpu()
    images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C
    h, w = images.size(0), images.size(1)

    # 1. un normalization
    images *= torch.Tensor([0.229, 0.224, 0.225])
    images += torch.Tensor([0.485, 0.456, 0.406])

    # 2. RGB to BGR
    image_np = images.numpy()

    # 3. box scaling
    bbox = results[0].cpu()
    cls = results[1].cpu()
    scores = results[2].cpu()

    ####################################
    # set threshold for visualization
    ####################################
    keep = scores > 0.1
    bbox = bbox[keep]
    cls = cls[keep]
    scores = scores[keep]
    ####################################

    plt.figure('result')
    plt.imshow(image_np)

    for i in range(len(bbox)):
        x1 = bbox[i][0].item() * w
        y1 = bbox[i][1].item() * h
        x2 = bbox[i][2].item() * w
        y2 = bbox[i][3].item() * h

        # class and score
        plt.text(x=x1 - 5,
                 y=y1 - 5,
                 s=label_array[label_dict[int(cls[i])]] + ' {:.2f}'.format(scores[i]),
                 fontsize=10,
                 bbox=dict(facecolor=color_array[int(cls[i])],
                           alpha=0.5))

        # bounding box
        plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                      width=x2 - x1,
                                      height=y2 - y1,
                                      linewidth=1,
                                      edgecolor=color_array[int(cls[i])],
                                      facecolor='none'))
    plt.show()


def test(epoch, vis, test_loader, model, criterion, opts, visualize=False):
    print('Testing of epoch [{}]'.format(epoch))
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),
                             map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            ## Get Loss!
            images = data[0]
            targets = data[1]
            images = images.to(device)
            outputs = model(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = criterion(outputs, targets)
            sum_loss += loss.item()

            ## Evaluate!
            pred_boxes, pred_labels, pred_scores = detect(pred=outputs)
            if opts.data_type == 'coco':
                img_id = test_loader.dataset.ids[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)
            else:
                print('not yet..')
                exit()

            evaluator.get_info(info)

            toc = time.time()

            # ---------- print ----------
            if idx % 1000 == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

            ## Visualize!
            if visualize:
                results = detect(outputs)
                visualize_results(images, results)

        mAP = evaluator.evaluate(test_loader.dataset)
        print('mAP for Epoch {} : {}'.format(epoch, mAP))
        print("Eval Time : {:.4f}".format(time.time() - tic))

        mean_loss = sum_loss / len(test_loader)
        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))

        # @@@ VISDOM @@@
        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))


if __name__ == "__main__":

    import os
    import sys
    import visdom
    from config import parse
    # import torchvision.transforms as T
    import dataset.transforms  as T
    from dataset.coco_dataset import COCO_Dataset
    from losses.hungarian_loss import HungarianLoss
    from losses.matcher import HungarianMatcher
    from models.detr import DETR

    # 1. configuration
    opts = parse(sys.argv[1:])

    # 2. visdom
    vis = None
    # if opts.data_root == "D:/data/coco":
    #     # for window
    #     vis = visdom.Visdom(port='8097')

    # 3. dataset
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_val = T.Compose([
        T.RandomResize([600], max_size=600),
        normalize,
    ])

    test_set = COCO_Dataset(root=opts.data_root,
                            split='val',
                            download=True,
                            transforms=transforms_val,
                            visualization=False)

    # 4. test loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    # 5. model (opts.num_classes = 91)
    model = DETR(num_classes=opts.num_classes, num_queries=100).to(device)
    if opts.distributed:
        model = torch.nn.DataParallel(model)

    # 6. criterion
    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=opts.num_classes, matcher=matcher).to(device)

    # 7. resume
    if opts.start_epoch != 0:
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1),
                                map_location=torch.device('cuda:{}'.format(0)))
        model.load_state_dict(checkpoint['model_state_dict'])  # load model state dict
        print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))

    else:
        print('\nNo check point to resume.. train from scratch.\n')

    # 11. test
    test(epoch=40,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         opts=opts,
         visualize=True,
         )


