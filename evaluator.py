import os
import json
import tempfile
from pycocotools.cocoeval import COCOeval


class Evaluator(object):
    def __init__(self, data_type='coco'):
        self.data_type = data_type
        if self.data_type == 'coco':
            self.results = list()
            self.img_ids = list()
        else:
            print("not ready yet..")
            exit()
        
    def get_info(self, info):
        if self.data_type=='coco':
            (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids) = info
            self.img_ids.append(img_id)

                        # convert coco_results coordination
            pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
            pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

            w = img_info['width']
            h = img_info['height']

            pred_boxes[:, 0] *= w
            pred_boxes[:, 2] *= w
            pred_boxes[:, 1] *= h
            pred_boxes[:, 3] *= h

            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                if int(pred_label) == 91:  # background label is 80     #FIXME Background 라벨 설정 디버깅 후 체크 필요
                    # print('background label :', int(pred_label))
                    continue

                coco_result = {
                    'image_id': img_id,
                    'category_id': coco_ids[int(pred_label)],     # FIXME 라벨 설정 필요 pred_label-1? pred_label?
                    'score': float(pred_score),
                    'bbox': pred_box.tolist(),
                }
                self.results.append(coco_result)

    def evaluate(self, dataset):
        if self.data_type == 'coco':
            _, tmp = tempfile.mkstemp()
            json.dump(self.results, open(tmp, "w"))

            cocoGt = dataset.coco
            cocoDt = cocoGt.loadRes(tmp)

            # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

            coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
            coco_eval.params.imgIds = self.img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            mAP = coco_eval.stats[0]
            mAP_50 = coco_eval.stats[1]
        
        return mAP