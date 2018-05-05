from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer.links.model.vision.resnet import BuildingBlock, ResNetLayers
from chainer import computational_graph as c
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
from chainercv.transforms.image.resize import resize
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import AnchorTargetCreator
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links.model.faster_rcnn.region_proposal_network import RegionProposalNetwork
from utils import ProposalTargetCreator
import roi_align_2d

class MaskRCNN(chainer.Chain):
    def __init__(self, extractor, rpn, head, mean,
                 min_size=600, max_size=1000,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 ):
        print("MaskRCNN initialization")
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('visualize')
    @property
    def n_class(self):
        return self.head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x) #VGG
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale) #Region Proposal Network
        roi_cls_locs, roi_scores, masks = self.head(
            h, rois, roi_indices) #Heads
        return roi_cls_locs, roi_scores, rois, roi_indices, masks

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def prepare(self, img):
        _, H, W = img.shape
        scale = self.min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

    def _suppress(self, raw_cls_bbox, raw_cls_roi, raw_prob, raw_mask):
        bbox = list()
        roi = list()
        label = list()
        score = list()
        mask = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            cls_roi_l = raw_cls_roi.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            lmask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[lmask]
            cls_roi_l = cls_roi_l[lmask]
            prob_l = prob_l[lmask]
            mask_l = raw_mask[:,l]
            mask_l = mask_l[lmask]
            keep = non_maximum_suppression(cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            roi.append(cls_roi_l[keep])
            #labels are in [0, self.nclass - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            mask.append(mask_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        roi = np.concatenate(roi, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.float32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        mask = np.concatenate(mask, axis=0).astype(np.float32)
        return bbox, roi,  label, score, mask

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        print("predicting!")
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append(size)
        bboxes = list()
        out_rois = list()
        labels = list()
        scores = list()
        masks = list()
        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                scale = img_var.shape[3] / size[1]
                roi_cls_locs, roi_scores, rois, _,  roi_masks = self.__call__(img_var, scale=scale)
           
            #assuming batch size = 1
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi_mask = F.sigmoid(roi_masks).data
            roi = rois / scale
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean), self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std), self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape).reshape((-1, 4))
            cls_bbox = loc2bbox(roi, roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
            cls_roi = roi.reshape((-1, self.n_class * 4))
            #clip the bbox
            cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
            cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])
            cls_roi[:, 0::2] = self.xp.clip(cls_roi[:, 0::2], 0, size[0])
            cls_roi[:, 1::2] = self.xp.clip(cls_roi[:, 1::2], 0, size[1])

            prob = F.softmax(roi_score).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_cls_roi = cuda.to_cpu(cls_roi)
            raw_prob = cuda.to_cpu(prob)
            raw_mask = cuda.to_cpu(roi_mask)
            bbox, out_roi, label, score, mask = self._suppress(raw_cls_bbox, raw_cls_roi, raw_prob, raw_mask)
            bboxes.append(bbox)
            out_rois.append(out_roi)
            labels.append(label)
            scores.append(score)
            masks.append(mask)

        return bboxes, out_rois, labels, scores, masks


class ExtractorResNet(ResNetLayers):
    def __init__(self, pretrained_model='auto', n_layers=50):
        print('ResNet',n_layers,' initialization')
        if pretrained_model=='auto':
            if n_layers == 50:
                pretrained_model = 'ResNet-50-model.caffemodel'
            elif n_layers == 101:
                pretrained_model = 'ResNet-101-model.caffemodel'
        super(ExtractorResNet, self).__init__(pretrained_model, n_layers)
        del self.fc6
        del self.res5
    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        return h

class MaskRCNNResNet(MaskRCNN):
    feat_stride = 16
    def __init__(self,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5 ,1, 2], anchor_scales=[8, 16, 32],
                 initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params=dict(),
                 roi_size=7,
                 n_layers=50, 
                 roi_align=True
                 ):
        print("MaskRNNResNet initialization")
        if n_fg_class is None:
            raise ValueError('supply n_fg_class!')
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if initialW is None:# and pretrained_model:
            print("setting initialW")
            initialW = chainer.initializers.Normal(0.01)
        self.roi_size=roi_size
        if pretrained_model is not None:
            pretrained_model = 'auto'
        extractor = ExtractorResNet(pretrained_model, n_layers=n_layers)
        rpn = RegionProposalNetwork(
            1024, 1024,
            ratios=ratios, anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = MaskRCNNHead(
            n_fg_class + 1,
            roi_size=self.roi_size, spatial_scale=1. / self.feat_stride,
            initialW=initialW, loc_initialW=loc_initialW, score_initialW=score_initialW,
            roi_align=roi_align
        )
        super(MaskRCNNResNet, self).__init__(
            extractor, rpn, head,
            mean=np.array([122.7717, 115.9465, 102.9801], dtype=np.float32)[:, None, None],
            min_size=min_size, max_size=max_size
        )

class MaskRCNNHead(chainer.Chain):
    def __init__(self, n_class, roi_size, spatial_scale,
                 initialW=None, loc_initialW=None, score_initialW=None, roi_align=True):
        super(MaskRCNNHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(3, 1024, 512, 2048, 1, initialW=initialW) 
            #class / loc branch
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)
            #Mask-RCNN branch
            self.deconvm1 = L.Deconvolution2D(2048, 256, 2, 2, initialW=initialW)
            self.convm2 = L.Convolution2D(256, n_class, 3, 1, pad=1,initialW=initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_align = roi_align
        print("ROI Align=",roi_align)

    def __call__(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        #x: (batch, channel, w, h)
        #rois: (128, 4) (ROI indices)
        if self.roi_align:
            pool = _roi_align_2d_yx(
                x, indices_and_rois, self.roi_size,self.roi_size,
                self.spatial_scale)
        else:
            pool = _roi_pooling_2d_yx(
                x, indices_and_rois, self.roi_size,self.roi_size,
                self.spatial_scale)

        #ROI, CLS  branch
        hres5 = self.res5(pool)
        fmap_size = hres5.shape[2:]
        h = F.average_pooling_2d(hres5, fmap_size, stride=1)
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)

        #Mask-RCNN branch
        h = F.relu(self.deconvm1(hres5)) 
        masks=self.convm2(h)
        return roi_cls_locs, roi_scores, masks

def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

def _roi_align_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = roi_align_2d.roi_align_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool


class MaskRCNNTrainChain(chainer.Chain):
    def __init__(self, mask_rcnn, rpn_sigma=3., roi_sigma=1., gamma=1,
                 anchor_target_creator=AnchorTargetCreator(),
                 roi_size=7):
        super(MaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.mask_rcnn = mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = ProposalTargetCreator(roi_size=roi_size)
        self.loc_normalize_mean = mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = mask_rcnn.loc_normalize_std
        self.decayrate=0.99
        self.avg_loss = None
        self.gamma=gamma
    def __call__(self, imgs, bboxes, labels, scale, masks):

        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        if isinstance(masks, chainer.Variable):
            masks = masks.data
        scale = np.asscalar(cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('only batch size 1 is supported')
        _, _, H, W = imgs.shape
        img_size = (H, W)
        #Extractor (VGG) : img -> features
        features = self.mask_rcnn.extractor(imgs)

        #Region Proposal Network : features -> rpn_locs, rpn_scores, rois
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.mask_rcnn.rpn(
            features, img_size, scale)
        bbox, label, mask, rpn_score, rpn_loc, roi = \
            bboxes[0], labels[0], masks[0], rpn_scores[0], rpn_locs[0], rois # batch size=1

        #proposal target : roi(proposed) , bbox(GT), label(GT) -> sample_roi, gt_roi_loc, gt_roi_label
        #the targets are compared with the head output.
        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = self.proposal_target_creator(
            roi, bbox, label, mask, self.loc_normalize_mean, self.loc_normalize_std)
        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)

        #Head Network : features, sample_roi -> roi_cls_loc, roi_score
        roi_cls_loc, roi_score, roi_cls_mask = self.mask_rcnn.head(
            features, sample_roi, sample_roi_index)

        #RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        #Head output losses
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
        roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label] 
        roi_mask = roi_cls_mask[self.xp.arange(n_sample), gt_roi_label]
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        #mask loss:  average binary cross-entropy loss
        mask_loss = F.sigmoid_cross_entropy(roi_mask[0:gt_roi_mask.shape[0]], gt_roi_mask)

        #total loss
        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + self.gamma * mask_loss

        #avg loss calculation
        if self.avg_loss is None:
            self.avg_loss = loss.data
        else:
            self.avg_loss = self.avg_loss * self.decayrate + loss.data*(1-self.decayrate)
        chainer.reporter.report({'rpn_loc_loss':rpn_loc_loss,
                                 'rpn_cls_loss':rpn_cls_loss,
                                 'roi_loc_loss':roi_loc_loss,
                                 'roi_cls_loss':roi_cls_loss,
                                 'roi_mask_loss':self.gamma * mask_loss,
                                 'avg_loss':self.avg_loss,
                                 'loss':loss}, self)
        return loss

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)
    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return F.sum(y)

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)
    in_weight = xp.zeros_like(gt_loc)
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss