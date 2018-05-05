import numpy as np
import json
import os
import cv2

from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, frPyObjects 

import chainer
from chainercv.utils import read_image

SBD_PATH = '/home/meet/datasets/VOC2012/benchmark_RELEASE/dataset/img/'

class VOCDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir='./', json_file='instances_train2017.json',
                 name='train2017', id_list_file='train2017.txt', sizemin=10):
        self.data_dir  = data_dir
        self.json_file = json_file
        self.coco = COCO(self.json_file)
        self.ids = [id_.strip() for id_ in open(data_dir+id_list_file)]
        self.name = name
        self.sizemin = sizemin

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        #i = i % 500 # for limiting data size
        numofboxes=0
        while True:
            id_ = self.ids[i]
            annot_labels, annot_bboxes, annot_segs= list(), list(), list()
            anno_ids = self.coco.getAnnIds(imgIds=[id_], iscrowd=None)
            annotations = self.coco.loadAnns(anno_ids)
            for a in annotations:
                if a['bbox'][2] > self.sizemin and a['bbox'][3] > self.sizemin \
                and a['iscrowd']==0 and a['category_id']<81:
                    annot_labels.append(a['category_id'])
                    annot_bboxes.append(a['bbox'])
                    annot_segs.append(a['segmentation'])
            numofboxes=len(annot_labels)
            if numofboxes > 0:
                break
            else:
                i = i - 1
        img_file = os.path.join(SBD_PATH, id_ + '.jpg')
        img = read_image(img_file, color=True)
        _, h, w = img.shape
        annot_bboxes = np.stack(annot_bboxes).astype(np.float32)
        annot_labels = np.stack(annot_labels).astype(np.int32)
        annot_masks, ii = [], 0
        for annot_seg_polygons in annot_segs:
            annot_masks.append(np.zeros((h, w), dtype=np.uint8))
            if isinstance(annot_seg_polygons, dict):
                decoded_segmentation = decode(annot_seg_polygons)
                offsety, offsetx, mh, mw = annot_bboxes[ii]    
                annot_masks[-1] = decoded_segmentation.astype(np.int32)
            ii += 1
        annot_masks = np.stack(annot_masks).astype(np.uint8) #y,x

        return img, annot_labels[0:ii], annot_bboxes[0:ii], annot_masks[0:ii], i
