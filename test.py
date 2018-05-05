
import os
import argparse
import chainer
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from utils import vis_bbox, visualize
from chainercv.datasets import voc_bbox_label_names
from maskrcnn import MaskRCNNResNet
from chainercv import utils

base_path = '/home/meet/datasets/VOC2012/benchmark_RELEASE/dataset/img/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--modelfile')
    args = parser.parse_args()

    model = MaskRCNNResNet(n_fg_class=20, roi_size=7, n_layers=50, roi_align=True, min_size=600, max_size=800)
    chainer.serializers.load_npz(args.modelfile, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    ids = [i.rstrip() for i in tuple(open('./test_sample.txt', 'r'))]
    count = 0

    flag = True
    while flag:
        count += 1
        if count == len(ids) -1:
            flag = False
        img_path = ids[count-1]
        print img_path
        #print("Please input full path to image here")
        #img_path = raw_input()
        if img_path == 'ex':
            flag = False
            return
        else:
            if 'jpg' in img_path or 'png' in img_path:
                fp = img_path
            else:
                fp = base_path  + img_path + '.jpg'
            if os.path.exists(fp):
                img = utils.read_image(fp, color=True)
                bboxes, rois, labels, scores, masks = model.predict([img])
                print(bboxes, rois)
                bbox, roi, label, score, mask = bboxes[0], rois[0], np.asarray(labels[0],dtype=np.int32), scores[0], masks[0]

                label_names=('background', 'aeroplane',  'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted_plant', 'sheep', 'sofa','train','tv_monitor')
                vis_bbox(
                    img, roi, roi, label=label, score=score, mask=mask, label_names=label_names, contour=False, labeldisplay=True)
                filename = "./static/{}_{}.png".format(args.modelfile[-8:-4], img_path)
                plt.savefig(filename)
            else:
                print('path given {} doesnt exist, pls verify'.format(fp))
        
if __name__ == '__main__':
    main()
