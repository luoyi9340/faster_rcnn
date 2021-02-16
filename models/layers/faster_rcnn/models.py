# -*- coding: utf-8 -*-  
'''
faster rcnn相关层

@author: luoyi
Created on 2021年2月9日
'''
import tensorflow as tf

import utils.conf as conf
from data.dataset import OriginalCrtBatchQueue
from data.dataset_rois import RoisCrtBatchQueue
from models.layers.rpn.preprocess import all_positives_from_fmaps
from models.layers.rpn.nms import nms


#    faster_rcnn layer
class FasterRcnnLayer(tf.keras.layers.Layer):

    def __init__(self,
                 cnns_layer=None,
                 rpn_layer=None,
                 pooling_layer=None,
                 fast_rcnn_layer=None,
                 
                 rpn_loss=None,
                 rpn_metrics_cls=None,
                 rpn_metrics_reg=None,
                 
                 faster_rcnn_loss=None,
                 faster_rcnn_metric_rpn_cls=None,
                 faster_rcnn_metric_rpn_reg=None,
                 
                 train_origina_queue=OriginalCrtBatchQueue.default(),
                 untrain_origina_queue=OriginalCrtBatchQueue.default(),
                 train_rpn_queue=RoisCrtBatchQueue.default(),
                 untrain_rpn_queue=RoisCrtBatchQueue.default(),
                 proposals_creator=None,
                 
                 threshold_rpn_prob=conf.RPN.get_nms_threshold_positives(),
                 threshold_rpn_nms_iou=conf.RPN.get_nms_threshold_iou(),
                 K=conf.ROIS.get_K(),
                 roi_areas=conf.ROIS.get_roi_areas(),
                 roi_scales=conf.ROIS.get_roi_scales(),
                 **kwargs):
        '''
           @param cnns_layer: 卷积层
           @param rpn_layer: rpn层
           @param pooling_layer: polling层
           @param fast_rcnn_layer: fast_rcnn层
           
           @param rpn_loss: rpn的loss对象
           @param rpn_metrics_cls: rpn的分类得分
           @param rpn_metrics_reg: rpn的回归得分
           
           @param faster_rcnn_loss: faster_rcnn的loss对象
           @param faster_rcnn_metric_rpn_cls: faster_rcnn的rpn分类得分
           @param faster_rcnn_metric_rpn_reg: faster_rcnn的rpn回归得分
           
           @param train_origina_queue:  暂存训练时的y（原生label）
           @param untrain_origina_queue: 暂存预测时的y（原生label）
           @param train_rpn_queue: 暂存训练时的y（rois）
           @param untrain_rpn_queue: 暂存预测时的y（rois）
           @param proposals_creator: proposals生成器
        '''
        super(FasterRcnnLayer, self).__init(name='FasterRcnnLayer', **kwargs)
        
        self._cnns_layer = cnns_layer
        self._rpn_layer = rpn_layer
        self._pooling_layer = pooling_layer
        self._fast_rcnn_layer = fast_rcnn_layer
        
        self._rpn_loss = rpn_loss
        self._rpn_metrics_cls = rpn_metrics_cls
        self._rpn_metrics_reg = rpn_metrics_reg
        
        self._faster_rcnn_loss = faster_rcnn_loss
        self._faster_rcnn_metric_rpn_cls = faster_rcnn_metric_rpn_cls
        self._faster_rcnn_metric_rpn_reg = faster_rcnn_metric_rpn_reg
        
        self._train_origina_queue = train_origina_queue
        self._untrain_origina_queue = untrain_origina_queue
        self._train_rpn_queue = train_rpn_queue
        self._untrain_rpn_queue = untrain_rpn_queue
        self._proposals_creator = proposals_creator
        
        self._threshold_rpn_prob = threshold_rpn_prob
        self._threshold_rpn_nms_iou = threshold_rpn_nms_iou
        self._K = K
        self._roi_areas = roi_areas
        self._roi_scales = roi_scales
        pass
    
    def call(self, x, training=None, **kwargs):
        '''输入 x=img, y=rois
            step1：过卷积层，拿到共享fmaps
            step2：过rpn层，拿到rpn层输出：tensor(batch_size, h, w, 6, K)
                    (batch_size, h, w, 0, K)    fmaps中[h, w]个像素点是前景的概率
                    (batch_size, h, w, 1, K)    fmaps中[h, w]个像素点是背景的概率
                    (batch_size, h, w, 2, K)    fmaps中[h, w]个像素点的d[x]
                    (batch_size, h, w, 3, K)    fmaps中[h, w]个像素点的d[y]
                    (batch_size, h, w, 4, K)    fmaps中[h, w]个像素点的d[w]
                    (batch_size, h, w, 5, K)    fmaps中[h, w]个像素点的d[h]
            step3：计算rpn层的loss和metrics，并记录
                    faster_rcnn_loss会记录rpn层的loss
                    faster_rcnn_metrics会记录rpn的cls和reg评价
            step4：通过rpn层输出产生proposals，作为后续fast_rcnn的y
            step5：proposals过pooling层，拿到统一尺寸的特征图
            step6：统一尺寸的特征图过fast_rcnn层，拿到fast_rcnn层输出：tensor(batch_size * num, 5, 分类数)
                    (batch_size * num, 0, 分类数)    每个proposal判定为对应分类的概率
                    (batch_size * num, 1, 分类数)    每个proposal判定为对应分类的d[x]
                    (batch_size * num, 2, 分类数)    每个proposal判定为对应分类的d[y]
                    (batch_size * num, 3, 分类数)    每个proposal判定为对应分类的d[w]
                    (batch_size * num, 4, 分类数)    每个proposal判定为对应分类的d[h]
            step7：计算fast_rcnn层的loss和metrics，并记录
                    faster_rcnn_loss会记录fast_rcnn层的loss
                    faster_rcnn_metrics会记录fast_rcnn的cls和reg评价
        '''
        #    step1：过cnn网络，计算共享特征图
        fmaps = self._cnns_layer(x)
        
        #    step2：过rpn网络，拿rpn输出
        y_pred = self._rpn_layer(fmaps)
        #    拿到y值（此时的y是原始y值）
        y_origina_true = self.get_y_origina(training)
        y_rpn_true = self.get_y_rois(training)
        
        #    step3：计算rpn loss和metrics
        rpn_loss = self._rpn_loss.call(y_rpn_true, y_pred)
        #    计算rpnloss
        self._faster_rcnn_loss.set_rpn_loss(rpn_loss)
        #    计算rpn的分类得分和回归得分，并暂存
        self._rpn_metrics_reg(y_rpn_true, y_pred)
        self._faster_rcnn_metric_rpn_cls.update_val(self._rpn_metrics_reg.result())
        self._faster_rcnn_metric_rpn_reg.update_val(self._rpn_metrics_reg.result())
        
        #    step4：用rpn的输出生成propoals
        anchors = all_positives_from_fmaps(fmaps, threshold=self._threshold_rpn_prob, K=self._K, roi_areas=self._roi_areas, roi_scales=self._roi_scales)
        anchors = nms(anchors, threshold=self._threshold_rpn_nms_iou)
        #    与当前的y值比较IoU，确定每个标签的训练数据
        self._proposals_creator.create_from_anchors()
        pass
    
    #    拿原始y值，并且x，y按照比例缩放好
    def get_y_origina(self, training=None):
        #    取y值
        if (training):
            y = self._train_origina_queue.crt_data()
            pass
        else:
            y = self._untrain_origina_queue.crt_data()
            pass
        return y

    #    拿rois_y值
    def get_y_rois(self, training=None):
        #    取y值
        if (training):
            y = self._train_rpn_queue.crt_data()
            pass
        else:
            y = self._untrain_rpn_queue.crt_data()
            pass
        return y

    pass


#    faster_rcnn特有的层，通过fmaps生成proposals
class CreateProposalsLayer(tf.keras.layers.Layer):

    def __init__(self, 
                 threshold_rpn_prob=conf.RPN.get_nms_threshold_positives(),
                 threshold_rpn_nms_iou=conf.RPN.get_nms_threshold_iou(),
                 
                 proposal_every_image=conf.PROPOSALES.get_proposal_every_image(),
                 **kwargs):
        '''
            @param threshold_rpn_prob: 判正的样本概率阈值（超过此阈值的概率值才被判为正样本）
            @param threshold_rpn_nms_iou: 非极大值抑制的IoU阈值（超过此阈值的anchor会被判重叠而过滤掉）
            
            @param proposal_every_image: 每张图片生成的proposal数量
            @return 
        '''
        self._threshold_rpn_prob = threshold_rpn_prob
        self._threshold_rpn_nms_iou = threshold_rpn_nms_iou
        
        super(CreateProposalsLayer, self).__init__(name='create_proposals_layer', **kwargs)
        pass

    #    前向
    def call(self, x, y_origina, **kwargs):
        '''
            @param x: rpn输出的fmaps
            @param y_origina: 原始的y数据 tensor(6, 5)
                                [vidx, x,y, w,h]    x,w经过缩放后的值
        '''
        
        pass

    pass

