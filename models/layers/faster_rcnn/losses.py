# -*- coding: utf-8 -*-  
'''
faster rcnn的联合loss
    loss = loss_rpn + loss_fast_rcnn

@author: luoyi
Created on 2021年2月9日
'''
import tensorflow as tf

import utils.conf as conf
from data.dataset_proposals import ProposalsCrtBatchQueue


#    faster rcnn 联合loss
class FasterRcnnLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 batch_size=conf.FASTER_RCNN.get_batch_size(),
                 num=conf.PROPOSALES.get_proposal_every_image(),
                 **kwargs):
        '''
            @param fast_rcnn_loss_layer: fast_rcnn的loss函数
        '''
        super(FasterRcnnLoss, self).__init__(**kwargs)
        
        #    rpn_loss
        self._rpn_loss = None
        #    fast_rcnn loss
        self._fast_rcnn_loss = None
        
        self._batch_size = batch_size
        self._num = num
        
        pass
    
    #    计算loss
    def call(self, y_true, y_pred):
        '''
            self._rpn_loss            tensor(batch_size, val)
            self._fast_rcnn_loss      tensor(batch_size * num, val)
        '''
        return self._rpn_loss + self._fast_rcnn_loss
    
    #    给rpn_loss
    def set_rpn_loss(self, rpn_loss):
        self._rpn_loss = rpn_loss
        pass
    #    给fast_rcnn loss
    def set_fast_rcnn_loss(self, fast_rcnn_loss):
        fast_rcnn_loss = tf.reshape(fast_rcnn_loss, shape=(self._batch_size, self._num, fast_rcnn_loss.shape[2]))
        fast_rcnn_loss = tf.math.reduce_mean(fast_rcnn_loss, axis=(1,2))
        self._fast_rcnn_loss = fast_rcnn_loss
        pass
    pass



#    计算rpn loss
class RpnLossLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 loss_layer=None,
                 rpn_loss_layer=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param loss_layer: faster_rcnn的loss对象
            @param rpn_loss_layer: rpn的loss对象
        '''
        super(RpnLossLayer, self).__init__(**kwargs)
        
        self._rpn_loss_layer = rpn_loss_layer
        self._loss_layer = loss_layer
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        pass
    
    #    前向
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算rpn loss
        rpn_loss = self._rpn_loss_layer.call(x, y)
        self._loss_layer.set_rpn_loss(rpn_loss)
        
        #    数据原样返回
        return x
    pass


#    fast_rcnn_loss层
class FastRcnnLossLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 loss_layer=None,
                 fast_rcnn_loss_layer=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param loss_layer: faster_rcnn的loss对象
            @param fast_rcnn_loss_layer: fast_rcnn的loss对象
        '''
        super(FastRcnnLossLayer, self).__init__(**kwargs)
        
        self._fast_rcnn_loss_layer = fast_rcnn_loss_layer
        self._loss_layer = loss_layer
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        pass
    
    #    前向
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算fast_rcnn loss
        fast_rcnn_loss = self._fast_rcnn_loss_layer.call(x, y)
        self._loss_layer.set_fast_rcnn_loss(fast_rcnn_loss)
        return x
    pass
