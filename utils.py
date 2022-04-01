import torch
import numpy as np

# set coco label color
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3)) / 255  # In plt, rgb color space's range from 0 to 1


def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)