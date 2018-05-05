import chainer
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from chainer.datasets import TransformDataset
from chainercv.datasets import VOCBboxDataset, voc_bbox_label_names
from chainercv import transforms
from chainercv.transforms.image.resize import resize

import argparse
import numpy as np
import time
from maskrcnn import MaskRCNNResNet
from voc_dataset import VOCDataset
from maskrcnn import MaskRCNNTrainChain
import logging
import traceback
from utils import SubDivisionUpdater
import cv2

def resize_bbox(bbox, in_size, out_size):
    bbox_o = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox_o[:, 0] = y_scale * bbox[:, 1]
    bbox_o[:, 2] = y_scale * (bbox[:, 1]+bbox[:, 3])
    bbox_o[:, 1] = x_scale * bbox[:, 0]
    bbox_o[:, 3] = x_scale * (bbox[:, 0]+bbox[:, 2])
    return bbox_o

train_annot_path = '/home/meet/datasets/VOC2012/benchmark_RELEASE/dataset/SBD_train2011.json'
val_annot_path = '/home/meet/datasets/VOC2012/benchmark_RELEASE/dataset/SBD_val2011.json'

def parse():
    parser = argparse.ArgumentParser(
        description='Mask RCNN trainer')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--batchsize', '-b', type=int, default=7)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=6000)
    parser.add_argument('--lr_step', '-ls', type=int, default=6000)
    parser.add_argument('--lr_initialchange', '-li', type=int, default=800)
    parser.add_argument('--pretrained', '-p', type=str, default='imagenet')
    parser.add_argument('--snapshot', type=int, default=1000)
    parser.add_argument('--validation', type=int, default=2000)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--iteration', '-i', type=int, default=10001)
    parser.add_argument('--roi_size', '-r', type=int, default=7, help='ROI size for mask head input')
    parser.add_argument('--gamma', type=float, default=1, help='mask loss weight')
    return parser.parse_args()

class Transform(object):
    def __init__(self, net):
        self.net = net
    def __call__(self, in_data):
        if len(in_data)==5:
            img, label, bbox, mask, i = in_data
        elif len(in_data)==4:
            img, bbox, label, i= in_data
        _, H, W = img.shape
        img = self.net.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        if len(bbox)==0:
            return img, [],[],1
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
        mask = resize(mask,(o_H, o_W))
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        mask = transforms.flip(mask, x_flip=params['x_flip'])
        return img, bbox, label, scale, mask

def main():
    args = parse()
    np.random.seed(args.seed)
    print('arguments: ', args)

    # Model setup
    train_data = VOCDataset(json_file=train_annot_path, name='train2011', id_list_file='train.txt')
    test_data = VOCDataset(json_file=val_annot_path, name='val2011', id_list_file='val.txt')    
    mask_rcnn = MaskRCNNResNet(n_fg_class=20, pretrained_model=args.pretrained,roi_size=args.roi_size, n_layers=50, roi_align = True, min_size=400, max_size=800)
    mask_rcnn.use_preset('evaluate')
    model = MaskRCNNTrainChain(mask_rcnn, gamma=args.gamma, roi_size=args.roi_size)
 
    # Trainer setup
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    train_data=TransformDataset(train_data, Transform(mask_rcnn))
    test_data=TransformDataset(test_data, Transform(mask_rcnn))
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    
    updater = SubDivisionUpdater(train_iter, optimizer, device=args.gpu, subdivisions=args.batchsize)
    
    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    # Extensions
    trainer.extend(
        extensions.snapshot_object(model.mask_rcnn, 'snapshot_model_{.updater.iteration}.npz'),
        trigger=(args.snapshot, 'iteration'))
    
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.lr_step, 'iteration'))
    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, model.mask_rcnn)
    log_interval = 40, 'iteration'
    plot_interval = 160, 'iteration'
    print_interval = 40, 'iteration'

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), trigger=(args.validation, 'iteration'))
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/avg_loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/roi_mask_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/loss',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1000))
    trainer.extend(extensions.dump_graph('main/loss'))
    try:
        trainer.run()
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()
