import torch
import torch.nn.functional as F
from torch import nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.backbone import Backbone
from models.transformer import Transformer


class DETR(nn.Module):
    def __init__(self, num_classes=81, num_queries=100):
        super().__init__()
        self.backbone = Backbone()
        self.transformer = Transformer()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.input_proj = nn.Conv2d(self.backbone.num_channels, 256, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, 256)
        self.class_embed = nn.Linear(256, num_classes + 1)
        self.bbox_embed = MLP(256, 256, 4, 3)
        self.aux_loss = True

    def forward(self, img):
        feat, pos = self.backbone(img)          # feat : (b,256,38,38) / pos : (b,256,38,38)
        feat2 = self.input_proj(feat)           # channel 2048 -> 256
        feat3 = self.transformer(feat2, self.query_embed.weight, pos)
        outputs_class = self.class_embed(feat3)
        outputs_coord = self.bbox_embed(feat3).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == '__main__':
    device_ids = [0]
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')

    model = DETR().to(device)

    image = torch.rand(2, 3, 600, 600).to(device)
    outputs = model(image)
    outputs['pred_logtis']
    print('test')
