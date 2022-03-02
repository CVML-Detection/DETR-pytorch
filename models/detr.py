import torch
import torch.nn.functional as F
from torch import nn

from .backbone import Backbone
from .transformer import Transformer


class DETR(nn.Module):
    def __init__(self, num_classes=81, num_queries=100):
        super().__init__()
        self.backbone = Backbone()
        self.transformer = Transformer()

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.input_proj = nn.Conv2d(backbone.num_channels, 256, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, 256)

    def forward(self, img):
        out, pos = self.backbone(img)
        out2 = self.input_proj(out)
        # feat = self.transformer(out2, self.query_embed)
        return 0


if __name__ == '__main__':
    device_ids = [0]
    device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')

    model = DETR().to(device)

    image = torch.rand(2, 3, 600, 600).to(device)
    out = model(image)
    print('test')