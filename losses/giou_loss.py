import torch
import torch.nn as nn


def giou_loss(boxes1, boxes2):
    """
    boxes1 [B, size, size, 3, 4]
    """
    # iou loss
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [2, s, s, 3]
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [2, s, s, 3]

    inter_left_up = torch.max(boxes1[:, None, :2], boxes2[..., :2])  # [B, s, s, 3, 2]
    inter_right_down = torch.min(boxes1[:, None, 2:], boxes2[..., 2:])  # [B, s, s, 3, 2]

    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))  # [B, s, s, 3, 2]
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area[:, None] + boxes2_area - inter_area  # [B, s, s, 3]
    ious = 1.0 * inter_area / union_area  # [B, s, s, 3]

    # iou_loss = 1 - ious
    # return iou_loss

    outer_left_up = torch.min(boxes1[:, None, :2], boxes2[..., :2])  # [B, s, s, 3, 2]
    outer_right_down = torch.max(boxes1[:, None, 2:], boxes2[..., 2:])  # [B, s, s, 3, 2]
    outer_section = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    outer_area = outer_section[..., 0] * outer_section[..., 1]  # [B, s, s, 3]

    giou = ious - (outer_area - union_area) / outer_area
    giou_loss = 1 - giou

    return giou_loss
