import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cxcy_to_xy


class HungarianLoss(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes  # class 갯수 - 이거 왜 90 개로 기준 되어있는것이지? -- issue?
        self.matcher = matcher

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #
        # exercise) _ indices : [[71, 0], [[15, 44], [1, 0]]] -> batch_idx : [0, 1, 1], src_inx : [71, 15, 44]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def giou_loss(self, boxes1, boxes2):
        """
        boxes1 [B, size, size, 3, 4]
        """
        # iou loss
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [2, s, s, 3]
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [2, s, s, 3]

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]

        inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))  # [B, s, s, 3, 2]
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area                                  # [B, s, s, 3]
        ious = 1.0 * inter_area / union_area                                                 # [B, s, s, 3]

        # iou_loss = 1 - ious
        # return iou_loss

        outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])                          # [B, s, s, 3, 2]
        outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])                       # [B, s, s, 3, 2]
        outer_section = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
        outer_area = outer_section[..., 0] * outer_section[..., 1]                           # [B, s, s, 3]

        giou = ious - (outer_area - union_area)/outer_area
        giou_loss = 1 - giou

        return giou_loss

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
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # 5) giou loss
        giou_loss = self.giou_loss(cxcy_to_xy(src_boxes), cxcy_to_xy(target_boxes))

        # # no mask loss
        # # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # losses = {}
        # losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        #
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        class_losses = loss_ce.sum()
        boxes_losses = loss_bbox.sum()
        giou_losses = giou_loss.sum()

        print("class losses : ", class_losses)
        print("boxes losses : ", boxes_losses)
        print("giou losses : ", giou_losses)
        total_loss = class_losses + boxes_losses + giou_losses
        return total_loss


if __name__ == '__main__':
    from losses.matcher import HungarianMatcher
    import torch

    # 7개의 category 가 있는 batch 2 의 targets
    targets = [{'boxes': torch.FloatTensor([[0.5012, 0.5481, 0.9976, 0.8812]]),
                'labels': torch.LongTensor([23]),
                'image_id': torch.LongTensor([285]),
                'area': torch.FloatTensor([264653.1875]),
                'iscrowd': torch.LongTensor([0]),
                'orig_size': torch.LongTensor([640, 586]),
                'size': torch.LongTensor([600, 600])},
               {'boxes': torch.FloatTensor([[0.6096, 0.5131, 0.3417, 0.8157],
                                           [0.6409, 0.8972, 0.6402, 0.0899]]),
                'labels': torch.LongTensor([1, 35]),
                'image_id': torch.LongTensor([785]),
                'area': torch.FloatTensor([36779.7031,  5123.4795]),
                'iscrowd': torch.LongTensor([0, 0]),
                'orig_size': torch.LongTensor([425, 640]),
                'size': torch.LongTensor([600, 600])}]

    # outputs_without_aux
    outputs = {'pred_logits': torch.randn([2, 100, 82]),
               'pred_boxes': torch.randn([2, 100, 4])
               }

    matcher = HungarianMatcher()
    criterion = HungarianLoss(num_classes=81, matcher=matcher)
    loss = criterion(outputs, targets)
    print(loss)