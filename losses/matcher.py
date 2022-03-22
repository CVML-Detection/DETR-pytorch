import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
# refer to https://github.com/facebookresearch/detr/blob/main/models/matcher.py


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()

        # according to official code
        self.cost_class = 1
        self.cost_bbox = 5
        self.cost_giou = 2

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: DETR네트워크의 아웃풋 cls, reg 의 값이 dict으로  :
                 "pred_logits": [batch_size, num_queries, num_classes]
                 "pred_boxes":  [batch_size, num_queries, 4]

            targets: 데이터셋의 getitem부분:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class # + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        split_cost_matrix = C.split(sizes, dim=-1)  # [(B, num_queries, objs of B1), (B, num_queries, objs of B2)]
        for b, cost_of_each_batch in enumerate(split_cost_matrix):
            c = cost_of_each_batch[b]
            indices.append(linear_sum_assignment(c))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


if __name__ == '__main__':
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
    indices = matcher(outputs, targets)
    print(indices)
    # e.g) [(tensor([78]), tensor([0])), (tensor([45, 63]), tensor([0, 1]))]
    # indices of batch 0 (row indices, col indices), indices of batch 1 (row indices, col indices)
