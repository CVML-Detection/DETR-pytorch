import torch
from config import device
import torch.nn.functional as F
from utils import box_cxcywh_to_xyxy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import coco_label_map as label_map
from utils import coco_color_array
from evaluator import Evaluator


def post_process(outputs, target_sizes):
    '''
    convert output of network to result(dictionary)
    :param outputs: outputs = {'pred_logits': ~ , 'pred_boxes': ~ }
    :param target_sizes:
    :return:
    '''
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results


def visualize_results(images, results):
    '''
    :param images:
    :param results: [{'scores': s, 'labels': l, 'boxes': b}]
    :param label_map:
    :param color_array:
    :return:
    '''
    label_array = list(label_map.keys())  # dict
    color_array = coco_color_array

    # 0. permute
    images = images.cpu()
    images = images.squeeze(0).permute(1, 2, 0)  # B, C, H, W --> H, W, C

    # 1. un normalization
    images *= torch.Tensor([0.229, 0.224, 0.225])
    images += torch.Tensor([0.485, 0.456, 0.406])

    # 2. RGB to BGR
    image_np = images.numpy()

    # 3. box scaling
    results = results[0]
    bbox = results['boxes'].cpu()
    cls = results['labels'].cpu()
    scores = results['scores'].cpu()

    plt.figure('result')
    plt.imshow(image_np)

    for i in range(len(bbox)):

        x1 = bbox[i][0]
        y1 = bbox[i][1]
        x2 = bbox[i][2]
        y2 = bbox[i][3]

        # class and score
        # plt.text(x=x1 - 5,
        #          y=y1 - 5,
        #          s=label_array[int(cls[i])] + ' {:.2f}'.format(scores[i]),
        #          fontsize=10,
        #          bbox=dict(facecolor=color_array[int(cls[i])],
        #                    alpha=0.5))

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
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch), map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    is_coco = hasattr(test_loader.dataset, 'coco')  # if True the set is COCO else VOC
    if is_coco:
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')

    evaluator = Evaluator(data_type=opts.data_type)

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get Loss!
            images = data[0]
            targets = data[1]
            images = images.to(device)
            outputs = model(images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = criterion(outputs, targets)

            # Evaluate!

            if visualize:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = post_process(outputs, orig_target_sizes)
                visualize_results(images, results)


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
    test(epoch=4,
         vis=vis,
         test_loader=test_loader,
         model=model,
         criterion=criterion,
         opts=opts,
         visualize=False,
         )


