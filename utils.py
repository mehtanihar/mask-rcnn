import cv2
import six
import copy

import numpy as np

from chainer import function, variable
from chainer import reporter, cuda
from chainer.training.updater import StandardUpdater
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_iou import bbox_iou
from chainercv.visualizations.vis_image import vis_image

from skimage.measure import find_contours
from matplotlib.patches import Polygon

class SubDivisionUpdater(StandardUpdater):
    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
        subdivisions=1, device=None, loss_func=None):
        super(SubDivisionUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            device=device,
            loss_func=loss_func,
        )
        self._batchsize = self._iterators['main'].batch_size
        self._subdivisions = subdivisions
        self._n = int(self._batchsize / self._subdivisions)
        assert self._batchsize % self._subdivisions == 0, (self._batchsize, self._subdivisions)

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays_list = []
        for i in range(self._subdivisions):
            in_arrays_list.append(self.converter(batch[i::self._subdivisions], self.device))
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        loss_func.cleargrads()

        losses=[]

        for i, in_arrays in enumerate(in_arrays_list):
            if isinstance(in_arrays, tuple):
                in_vars = list(variable.Variable(x) for x in in_arrays)
                loss = loss_func(*in_vars)
                losses.append(loss)
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x) for key, x in six.iteritems(in_arrays)}
                loss = loss_func(in_vars)
                losses.append(loss)
            else:
                print(type(in_arrays))
            loss.backward()
        
        optimizer.update()
        if isinstance(loss, dict):
            avg_loss = {k: 0. for k in losses[0].keys()}
            for loss in losses:
                for k, v in loss.items():
                    avg_loss[k] += v
            #avg_loss = {k: v / float(self._batchsize) for k, v in avg_loss.items()}
            avg_loss = {k: v / float(len(losses)) for k, v in avg_loss.items()}
            #avg_loss = {k: v for k, v in avg_loss.items()}

            for k, v in avg_loss.items():
                reporter.report({k: v}, loss_func)
            reporter.report({'loss': sum(list(avg_loss.values()))}, loss_func)
        else:
            avg_loss = 0.
            for loss in losses:
                avg_loss += loss
            reporter.report({'loss': avg_loss}, loss_func)


class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0,
                 roi_size=7
                 ):

        self.roi_size=roi_size
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label, mask,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)
        mask = cuda.to_cpu(mask)

        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        
        gt_roi_mask=[]
        _, h, w = mask.shape
        for i , idx in enumerate(gt_assignment[pos_index]):
            A=mask[idx, np.max((int(sample_roi[i,0]),0)):np.min((int(sample_roi[i,2]),h)), np.max((int(sample_roi[i,1]),0)):np.min((int(sample_roi[i,3]),w))]
            gt_roi_mask.append(cv2.resize(A, (self.roi_size*2,self.roi_size*2)))

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_loc = cuda.to_gpu(gt_roi_loc)
            gt_roi_label = cuda.to_gpu(gt_roi_label) 
            gt_roi_mask = cuda.to_gpu(np.stack(gt_roi_mask).astype(np.int32))
        else:
            gt_roi_mask = np.stack(gt_roi_mask).astype(np.int32)
        return sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask


def vis_bbox(img, bbox, roi, label=None, score=None, mask=None, label_names=None, ax=None, contour=False, labeldisplay=True):

    from matplotlib import pyplot as plot

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax
    COLOR=[(1,1,0), (1,0,1),(0,1,1),(0,0,1),(0,1,0), (1,0,0),(0.1,1,0.2)]

    for i, (bb, r) in enumerate(zip(bbox, roi)):
        #print(label[i])
        #if label[i] >1:
        #    continue
        xy = (bb[1], bb[0])
        height = int(bb[2]) - int(bb[0])
        width = int(bb[3]) - int(bb[1])
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=1))
        if mask is not None:
            M=mask[i]
            padded_mask = np.zeros((img.shape[2], img.shape[1]), dtype=np.uint8)
            resized_mask = cv2.resize(mask[i].T*255,(height, width))
            padded_mask[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = resized_mask
            Mcontours = find_contours(padded_mask/255, 0.3)
            for verts in Mcontours:
                p = Polygon(verts, facecolor="none", edgecolor=[1,1,1])
                
        #print(M)
        caption = list()
        for my in range(14):
            for mx in range(14):
                mxy = (r[1]+(r[3]-r[1])/14*mx, r[0]+(r[2]-r[0])/14*my)
                Mcolor=np.clip((M[my,mx])*1,0,0.5)
                #print(Mcolor)
                ax.add_patch(plot.Rectangle(mxy, int((r[3]-r[1])/14)+1,int((r[2]-r[0])/14)+1,
                fill=True, linewidth=0,facecolor=COLOR[i%len(COLOR)], alpha=Mcolor))
                if contour:
                    ax.add_patch(p)
        if label is not None and label_names is not None:
            lb = label[i]
            print(lb)
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0 and labeldisplay:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax

def visualize(im, labels, bboxes, masks, scale=1.0, show=False, fullSizeMask=False):
    """
    visualize all detections in one image
    :param im: np.ndarray with shape [b=1 c h w] in rgb in 0-255
    :param labels: list of ids of instances in the im
    :param bboxes: [numpy.ndarray([[x1 y1 x2 y2]]) for j in len(labels)]
    :param masks: [numpy.ndarray with shape [h, w] for j in len(labels)]
    :param masks: [numpy.ndarray with shape [h, w] for j in len(labels)]

    :return im: np.ndarray of final image with visualizations that can be saved 
    """
    import matplotlib.pyplot as plt
    plt.cla()
    plt.axis("off")

    im = np.expand_dims(im, 0)
    im = im[0].transpose(1,2,0) / 255.0 
    im = im.astype(np.float64)
    plt.imshow(im)

    id2cls = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
              6: 'bus', 7: 'car ', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
              12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'potted_plant',
              17: 'sheep', 18: 'sofa', 19:'train', 20: 'tv_monitor'}

    pascal_labels = np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                        [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                        [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                        [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                        [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                        [0,64,128]])

    for j, _id in enumerate(labels):
        name = id2cls[_id]
        if name == '__background__':
            continue
        det = bboxes[j]
        msk = masks[j]
        color = pascal_labels[_id] / 255.0
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