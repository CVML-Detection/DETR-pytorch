import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from config import device
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import platform
import random
import os
import wget
import glob
import zipfile
# from utils import bar_custom, coco_color_array
from dataset.detection_transforms import mosaic
import dataset.transforms as T
from dataset.transforms import ConvertCocoTarget


def download_coco(root_dir='D:\data\\coco', remove_compressed_file=True):
    # for coco 2017
    coco_2017_train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    coco_2017_val_url = 'http://images.cocodataset.org/zips/val2017.zip'
    coco_2017_test_url = 'http://images.cocodataset.org/zips/test2017.zip'
    coco_2017_trainval_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    os.makedirs(root_dir, exist_ok=True)

    img_dir = os.path.join(root_dir, 'images')
    anno_dir = os.path.join(root_dir, 'annotations')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    """Download the VOC data if it doesn't exit in processed_folder already."""

    # if (os.path.exists(os.path.join(img_dir, 'train2017')) and
    #         os.path.exists(os.path.join(img_dir, 'val2017')) and
    #         os.path.exists(os.path.join(img_dir, 'test2017'))):
    #
    if (os.path.exists(os.path.join(img_dir, 'train2017')) and
            os.path.exists(os.path.join(img_dir, 'val2017'))):

        print("Already exist!")
        return

    print("Download...")

    # image download
    # wget.download(url=coco_2017_train_url, out=img_dir, bar=bar_custom)
    print('')
    # wget.download(url=coco_2017_val_url, out=img_dir, bar=bar_custom)
    print('')
    # wget.download(url=coco_2017_test_url, out=img_dir, bar=bar_custom)
    # print('')

    # annotation download
    wget.download(coco_2017_trainval_anno_url, out=root_dir, bar=bar_custom)
    print('')

    print("Extract...")

    # image extract
    with zipfile.ZipFile(os.path.join(img_dir, 'train2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    with zipfile.ZipFile(os.path.join(img_dir, 'val2017.zip')) as unzip:
        unzip.extractall(os.path.join(img_dir))
    # with zipfile.ZipFile(os.path.join(img_dir, 'test2017.zip')) as unzip:
    #     unzip.extractall(os.path.join(img_dir))

    # annotation extract
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))

    # remove zips
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))  # in root_dir remove *.zip
        for anno_zip in root_zip_list:
            os.remove(anno_zip)

        img_zip_list = glob.glob(os.path.join(img_dir, '*.zip'))  # in img_dir remove *.zip
        for img_zip in img_zip_list:
            os.remove(img_zip)
        print("Remove *.zips")

    print("Done!")


# COCO_Dataset
class COCO_Dataset(Dataset):
    def __init__(self,
                 root='D:\Data\coco',
                 split='train',
                 download=True,
                 transforms=None,
                 visualization=False):
        super().__init__()

        if platform.system() == 'Windows':
            matplotlib.use('TkAgg')  # for window

        # -------------------------- set root --------------------------
        self.root = root

        # -------------------------- set split --------------------------
        assert split in ['train', 'val', 'test']
        self.split = split
        self.set_name = split + '2017'

        # -------------------------- download --------------------------
        self.download = download
        if self.download:
            download_coco(root_dir=root)

        # -------------------------- transform --------------------------
        self.transforms = transforms

        # -------------------------- visualization --------------------------
        self.visualization = visualization

        self.img_path = glob.glob(os.path.join(self.root, 'images', self.set_name, '*.jpg'))
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_' + self.set_name + '.json'))

        self.ids = list(sorted(self.coco.imgToAnns.keys()))  # anno 가 존재하는 것만

        self.coco_ids = sorted(self.coco.getCatIds())  # list of coco labels [1, ...11, 13, ... 90]  # 0 ~ 79 to 1 ~ 90
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)}  # 1 ~ 90 to 0 ~ 79
        # int to int
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in
                                        self.coco.loadCats(self.coco_ids)}  # len 80

        self.prepare = ConvertCocoTarget()
        # int to string
        # {1 : 'person', 2: 'bicycle', ...}
        '''
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        '''

    def __getitem__(self, index):
        # 1. id
        id = self.ids[index]

        # 2. load image
        path = self.coco.loadImgs(ids=id)[0]['file_name']
        image = Image.open(os.path.join(self.root, 'images', self.set_name, path)).convert('RGB')
        # eg. 'D:\\Data\\coco\\images\\val2017\\000000289343.jpg'
        #     |      root     |images |set_name|     path       |

        # 3. load anno
        target = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=id))
        target = {'image_id': id, 'annotations': target}

        # 4. convert to coco target
        image, target = self.prepare(image, target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def collate_fn(self, batch):

        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        images = torch.stack(images, dim=0)
        return images, targets

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':

    import torchvision.transforms as transforms
    # import dataset.detection_transforms as det_transforms

    # transform_train = det_transforms.DetCompose([
    #     # ------------- before Tensor augmentation -------------
    #     det_transforms.DetRandomPhotoDistortion(),
    #     det_transforms.DetRandomHorizontalFlip(),
    #     det_transforms.DetToTensor(),
    #     # ------------- for Tensor augmentation -------------
    #     det_transforms.DetRandomZoomOut(max_scale=3),
    #     det_transforms.DetRandomZoomIn(),
    #     det_transforms.DetResize(size=600, max_size=1000, box_normalization=True),
    #     # det_transforms.DetRandomSizeCrop(384, 600),  # FIXME - only if box_normalization=True
    #     det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # ])
    #
    # transform_test = det_transforms.DetCompose([
    #     det_transforms.DetToTensor(),
    #     det_transforms.DetResize(size=(600, 600), box_normalization=True),
    #     det_transforms.DetNormalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # ])

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_val = T.Compose([
        T.RandomResize([600, 600], max_size=600),
        normalize,
    ])

    coco_dataset = COCO_Dataset(root="D:/data/coco",
                                split='val',
                                download=True,
                                transforms=transforms_val,
                                visualization=True)

    # coco_dataset = COCO_Dataset(root="D:/data/coco",
    #                             split='train',
    #                             download=True,
    #                             transform=transform_train,
    #                             visualization=True)

    val_loader = torch.utils.data.DataLoader(coco_dataset,
                                             batch_size=2,
                                             collate_fn=coco_dataset.collate_fn,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True)

    for i, data in enumerate(val_loader):

        images = data[0]
        targets = data[1]

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(targets)
