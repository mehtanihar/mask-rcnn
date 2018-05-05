import os
import glob
import cv2
import json
import torch
import random
import collections
import numpy as np
import scipy.misc as m
import scipy.io as io

import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import ndimage
from torch.utils import data
from os.path import join as pjoin

from pycocotools.mask import encode, decode, frPyObjects 
from pycocotools.coco import COCO

SBD_PATH = '/home/meet/datasets/VOC2012/benchmark_RELEASE/dataset/'

class SBDLoader(data.Dataset):
    """Data loader for the Extended Pascal VOC (SBD) dataset.

    Annotations from both the the SBD (Berkeley) dataset (where annotations
    are stored as .mat files) are converted into a common `label_mask` 
    format.  Under this format, each mask is an (M,N) array of integer values
     from 0 to 21, where 0 represents the background class.
    
    We sample images which have segmentation as well detection annotations 
    available and convert them to MS COCO Format for faster retrival at train
    time and to optimize disk space for saved annotations. 
    """

    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 20
        self.sizemin = 10
        self.id2cls = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
                       6: 'bus', 7: 'car ', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                       12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted_plant',
                       17: 'sheep', 18: 'sofa', 19:'train', 20: 'tv_monitor'}
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.splits = collections.defaultdict(list)
        
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)

        self.setup_annotations()
        self.coco = COCO(os.path.join(self.root, 'SBD_{}2011.json'.format(self.split)))

    def __len__(self):
        return len(self.splits[self.split])


    def __getitem__(self, index):
        n_boxes = 0
        while True:
            id_ = self.splits[self.split][index]
            annot_labels, annot_bboxes, annot_segs= list(), list(), list()
            anno_ids = self.coco.getAnnIds(imgIds=[id_], iscrowd=None)
            annotations = self.coco.loadAnns(anno_ids)
            for a in annotations:
                if a['bbox'][2] > self.sizemin and a['bbox'][3] > self.sizemin \
                and a['iscrowd'] == 0 and a['category_id'] < 21:
                    annot_labels.append(a['category_id'])
                    annot_bboxes.append(a['bbox'])
                    annot_segs.append(a['segmentation'])
            n_boxes = len(annot_labels)
            if n_boxes > 0:
                break
            else:
                i = i - 1

        img_path = os.path.join(self.root, 'img', id_ + '.jpg')
        img = m.imread(img_path).transpose((2, 0, 1))
        channels, h, w = img.shape
        annot_bboxes = np.stack(annot_bboxes).astype(np.float32)
        annot_labels = np.stack(annot_labels).astype(np.int32)
        annot_masks, ii = [], 0
        for annot_seg_polygons in annot_segs:
            annot_masks.append(np.zeros((h, w), dtype=np.uint8))

            if isinstance(annot_seg_polygons, dict):
                #print("What a dic")
                decoded_segmentation = decode(annot_seg_polygons)
                offsety, offsetx, mh, mw = annot_bboxes[ii]    
                annot_masks[-1] = decoded_segmentation.astype(np.int32)
            else:
                mask_in = np.zeros((int(h), int(w)), dtype=np.uint8) #(y,x)
                offsety, offsetx, mh, mw = annot_bboxes[ii]
                for annot_seg_polygon in annot_seg_polygons:
                    N = len(annot_seg_polygon)
                    rr, cc = polygon(np.array(annot_seg_polygon[1:N:2]), np.array(annot_seg_polygon[0:N:2]))
                    mask_in[np.clip(rr,0,h-1), np.clip(cc,0,w-1)] = 1. #y, x
                annot_masks[-1] = np.asarray(mask_in, dtype=np.int32)
            ii += 1
        annot_masks = np.stack(annot_masks).astype(np.uint8) #y,x
        return {'img': img,
                'lbls': annot_labels[0:ii], 
                'bbox': annot_bboxes[0:ii],
                'masks': annot_masks[0:ii], 
                'idx': index }


    def dict2instance(self, data):
        return data['img'], data['lbls'], data['bbox'], data['masks'], data['idx']


    def get_bbox_and_area(self, binary_mask):
        slice_x, slice_y = ndimage.find_objects(binary_mask)[0]
        return map(int, [slice_y.start,
                         slice_x.start,
                         slice_y.stop - slice_y.start,
                         slice_x.stop - slice_x.start]), float(np.sum(binary_mask))


    def get_rle_seg(self, binary_mask):
        rle = encode(np.array(binary_mask, dtype=np.uint8, order='F'))
        return rle


    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                          [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                          [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                          [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                          [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                          [0,64,128]])


    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def get_baseCOCO_dict(self):
        """Returns a base dictionary for storing annotations in MS COCO
           format
        
        Returns:
            dict: The base dictionary with meta information
        """ 

        base_json = {}
        base_json['type'] = 'instances'
        base_json['info'] = {u'contributor': u'SBD Berkeley Vision',
                             u'year': 2011}

        base_json['categories'] = [{u'id': k, 'name': self.id2cls[k] , 'supercategory': self.id2cls[k]} \
                                   for k in sorted(self.id2cls.keys())]        
        return base_json


    def setup_annotations(self):

        train_path = pjoin(self.root, 'train.txt')
        val_path = pjoin(self.root, 'val.txt')

        sbd_train_list = [id_.rstrip() for id_ in tuple(open(train_path, 'r'))]
        sbd_val_list = [id_.rstrip() for id_ in tuple(open(val_path, 'r'))]

        split_dic = {'SBD_train2011.json': sbd_train_list,
                     'SBD_val2011.json': sbd_val_list,}

        # If Annotations already exists in COCO format 
        if os.path.exists(pjoin(self.root, 'SBD_train2011.json')) and \
           os.path.exists(pjoin(self.root, 'SBD_val2011.json')):
            print("Using COCO formatted annotations already cached at {}".format(self.root))
            self.splits['train'] = sbd_train_list
            self.splits['val'] = sbd_val_list

        else:
            print("Caching annotations in COCO format at {}".format(self.root))
            for k, split_list in split_dic.items():
                self.coco_dic = self.get_baseCOCO_dict()
                self.coco_dic['images'] = []
                self.coco_dic['annotations'] = []

                annotation_count = 0
                for idx, img_id in tqdm(enumerate(split_list)):
                    img_path = os.path.join(self.root, 'img', img_id + '.jpg')
                    ann_path = os.path.join(self.root, 'inst', img_id + '.mat')
                    img = m.imread(img_path)
                    self.coco_dic['images'].append({u'file_name': img_path,
                                                    u'height': img.shape[0],
                                                    u'width': img.shape[1],
                                                    u'id': img_id})
                    
                    ann = io.loadmat(ann_path)
                    cls_present = ann['GTinst'][0]['Categories'][0]
                    for instance in range(1,len(cls_present)+1):
                        segmentation = ann['GTinst'][0]['Segmentation'][0] == instance
                        bbox, area = self.get_bbox_and_area(segmentation)
                        rle_seg = self.get_rle_seg(segmentation)
                        self.coco_dic['annotations'].append({u'area': area,
                                                            u'bbox': bbox,
                                                            u'category_id': int(cls_present[instance-1]),
                                                            u'id': annotation_count,
                                                            u'image_id': img_id,
                                                            u'iscrowd': 0,
                                                            u'segmentation': rle_seg})
                        annotation_count += 1
                
                save_path = os.path.join(self.root, k)
                print("Saved {} annotations in COCO format at {}".format(annotation_count, save_path))
                with open(save_path, 'w') as outfile:
                    json.dump(self.coco_dic, outfile)

    def visualize(self, im, labels, bboxes, masks, scale=1.0, show=False, fullSizeMask=False):
        """
        visualize all detections in one image
        :param im: np.ndarray with shape [b=1 c h w] in rgb in 0-255
        :param labels: list of ids of instances in the im
        :param bboxes: [numpy.ndarray([[x1 y1 x2 y2]]) for j in len(labels)]
        :param masks: [numpy.ndarray with shape [h, w] for j in len(labels)]
        :param masks: [numpy.ndarray with shape [h, w] for j in len(labels)]

        :return im: np.ndarray of final image with visualizations that can be saved 
        """
        plt.cla()
        plt.axis("off")
        im = im[0].transpose(1,2,0) / 255.0 
        im = im.astype(np.float64)
        plt.imshow(im)

        for j, _id in enumerate(labels):
            name = self.id2cls[_id]
            if name == '__background__':
                continue
            det = bboxes[j]
            msk = masks[j]
            color = self.get_pascal_labels()[_id] / 255.0
            bbox = det[:4] * scale
            cod = bbox.astype(int)
            if not fullSizeMask:
                if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:
                    msk = cv2.resize(msk, im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, 0].T.shape)
                    bimsk = msk >= 0.5
                    bimsk = bimsk.astype(int)
                    bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                    mskd = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] * bimsk
                    clmsk = np.ones(bimsk.shape) * bimsk
                    clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                    clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                    clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                    im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] + 0.8 * clmsk - 0.8 * mskd
            else:
                bimsk = msk >= 0.5
                bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                clmsk = np.ones(bimsk.shape) * bimsk
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0]
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1]
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2]
                im = 0.7 * im + 0.3 * clmsk
            score = 0.8
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                                fill=False, edgecolor=color, linewidth=3))
            plt.gca().text((bbox[2]+bbox[0])/2, bbox[1],
                            '{:s} {:.3f}'.format(name, score),
                            bbox=dict(facecolor=color, alpha=0.9), fontsize=8, color='white')
        plt.imshow(im)
        if show:
            plt.show()
        return im


if __name__ == '__main__':
    bs = 1
    dst = SBDLoader(root=SBD_PATH, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=bs, shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, labels, bboxes, masks, index = dst.dict2instance(data)
        dst.visualize(imgs.numpy(), labels.numpy()[0], bboxes.numpy()[0], masks.numpy()[0], show=True, fullSizeMask=True)
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()