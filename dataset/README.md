# Dataset of DETR-cvml 

### Data structure 

- coco_dataset.py
- detection_transforms.py
- voc_dataset.py

### Data transforms (augmentation) 

- [x] Resize
- [x] Zoom In
- [x] Zoom Out 
- [x] Photometric Distort
- [x] Horizontal Flip
- [x] Mosiac

### Analysis data structure

- [x] Data structure of official DETR
```
target = 
{'boxes' : tensor([# obj, 4], float32),
 'labels' : tensor([#obj]), int64),
 'image_id' : tensor([1]), int64),
 'area' : tensor([#obj]), float32),
 'iscrowd' : tensor([#obj - 0 or 1], int64),
 'orig_size' : tensor([2 - H x W], int64),
 'size' : tensor([2 - H x W], int64)}
```
- [x] batch targets and collate_fn
```
--------------------------------------------------------------------
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
--------------------------------------------------------------------
def collate_fn(self, batch):

    images = list()
    targets = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images, dim=0)
    return images, targets
--------------------------------------------------------------------
``` 
- [ ] visualize images
- [ ] run out the transforms (random resize -> normal resize)
