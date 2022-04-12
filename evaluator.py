import os

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
            print('...')
    
    def evaluate(self, dataset):
        if self.data_type=='coco':
            mAP = 0
        
        return mAP