import torch.nn as nn
import torch.nn.functional as F
from utils import cxcy_to_xy, box_cxcywh_to_xyxy
from losses.generalized_iou import generalized_box_iou
from torch import Tensor
import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class HungarianLoss(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes  # class 갯수 - 이거 왜 90 개로 기준 되어있는것이지? -- issue?
        self.matcher = matcher

        self.eos_coef = 0.1
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', self.empty_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #
        # exercise) _ indices : [[71, 0], [[15, 44], [1, 0]]] -> batch_idx : [0, 1, 1], src_inx : [71, 15, 44]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        # 1) get permutation
        indices = self.matcher(outputs, targets)

        # 2) convert to permutation for src
        src_idx = self._get_src_permutation_idx(indices)

        # 3) cls loss
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[src_idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)

        # 4) box loss
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'][src_idx]
        num_boxes = src_boxes.size(0)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # 5) giou loss
        giou_loss = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )

        class_losses = loss_ce
        boxes_losses = loss_bbox.sum() / num_boxes
        giou_losses = giou_loss.sum() / num_boxes

        print("class losses : ", class_losses)
        print("boxes losses : ", boxes_losses)
        print("giou losses : ", giou_losses)

        total_loss = 1 * class_losses + 1 * boxes_losses + 1 * giou_losses
        return total_loss


if __name__ == '__main__':
    from losses.matcher import HungarianMatcher
    import torch

    targets = [{'boxes': torch.FloatTensor([[0.3896, 0.4161, 0.0386, 0.1631],
                                            [0.1276, 0.5052, 0.2333, 0.2227],
                                            [0.9342, 0.5835, 0.1271, 0.1848],
                                            [0.6047, 0.6325, 0.0875, 0.2414],
                                            [0.5025, 0.6273, 0.0966, 0.2312],
                                            [0.6692, 0.6190, 0.0471, 0.1910],
                                            [0.5128, 0.5283, 0.0337, 0.0272],
                                            [0.6864, 0.5320, 0.0829, 0.3240],
                                            [0.6125, 0.4462, 0.0236, 0.0839],
                                            [0.8119, 0.5017, 0.0230, 0.0375],
                                            [0.7863, 0.5364, 0.0317, 0.2542],
                                            [0.9562, 0.7717, 0.0224, 0.1073],
                                            [0.9682, 0.7781, 0.0201, 0.1090],
                                            [0.7106, 0.3100, 0.0218, 0.0514],
                                            [0.8866, 0.8316, 0.0573, 0.2105],
                                            [0.5569, 0.5167, 0.0178, 0.0529],
                                            [0.6517, 0.5288, 0.0150, 0.0294],
                                            [0.3880, 0.4784, 0.0222, 0.0414],
                                            [0.5338, 0.4879, 0.0152, 0.0393],
                                            [0.6000, 0.6471, 0.1962, 0.2088]]),
                 'labels': torch.LongTensor([64, 72, 72, 62, 62, 62, 62,  1,  1, 78, 82, 84, 84, 85, 86, 86, 62, 86, 86, 67]),
                 'image_id': torch.LongTensor([139]),
                 'area': torch.FloatTensor([702.2101, 17488.5430,  7702.1807,  2964.8022,  2421.3699,  1702.5177,
                                            277.4844,  3846.5366,   574.5752,   287.4813,  2759.6499,   447.1068,
                                            425.9598,   297.9721,  2867.4546,   235.2796,   120.1416,   250.2994,
                                            158.7570,  3119.4846]),
                 'iscrowd': torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 'orig_size': torch.LongTensor([426, 640]),
                 'size': torch.LongTensor([600, 600])},

                {'boxes': torch.FloatTensor([[0.5012, 0.5481, 0.9976, 0.8812]]),
                 'labels': torch.LongTensor([23]),
                 'image_id': torch.LongTensor([285]),
                 'area': torch.FloatTensor([264653.1875]),
                 'iscrowd': torch.LongTensor([0]),
                 'orig_size': torch.LongTensor([640, 586]),
                 'size': torch.LongTensor([600, 600])}]

    # 7개의 category 가 있는 batch 2 의 targets
    # targets = [{'boxes': torch.FloatTensor([[0.5012, 0.5481, 0.9976, 0.8812]]),
    #             'labels': torch.LongTensor([23]),
    #             'image_id': torch.LongTensor([285]),
    #             'area': torch.FloatTensor([264653.1875]),
    #             'iscrowd': torch.LongTensor([0]),
    #             'orig_size': torch.LongTensor([640, 586]),
    #             'size': torch.LongTensor([600, 600])},
    #            {'boxes': torch.FloatTensor([[0.6096, 0.5131, 0.5417, 0.8157],
    #                                        [0.6409, 0.8972, 0.6402, 0.8099]]),
    #             'labels': torch.LongTensor([1, 35]),
    #             'image_id': torch.LongTensor([785]),
    #             'area': torch.FloatTensor([36779.7031,  5123.4795]),
    #             'iscrowd': torch.LongTensor([0, 0]),
    #             'orig_size': torch.LongTensor([425, 640]),
    #             'size': torch.LongTensor([600, 600])}]

    # outputs_without_aux
    torch.manual_seed(1)
    outputs = {'pred_logits': torch.randn([2, 100, 92]),
               'pred_boxes': torch.sigmoid(torch.randn([2, 100, 4]))
               }

    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=91, matcher=matcher)
    loss = criterion(outputs, targets)
    print(loss)