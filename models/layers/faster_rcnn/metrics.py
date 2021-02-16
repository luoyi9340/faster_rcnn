# -*- coding: utf-8 -*-  
'''
faster rcnn评价指标
    metrics = metrics_rpn ∪ metrics_fast_rcnn

@author: luoyi
Created on 2021年2月10日
'''
import tensorflow as tf

from data.dataset_proposals import ProposalsCrtBatchQueue


############################################################################################################
#    rpn层评价函数
############################################################################################################
#    rpn分类评价层
class RpnMetricsClsLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 metrics_layer=None,
                 rpn_metrics_cls=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param metrics_layer: faster_rcnn的评价对象
            @param rpn_metrics_cls: rpn的分类评价对象
        '''
        super(RpnMetricsClsLayer, self).__init__(name='RpnMetricsClsLayer', **kwargs)
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        
        self._metrics_layer = metrics_layer
        self._rpn_metrics_cls = rpn_metrics_cls
        pass
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算rpn的分类损失
        self._rpn_metrics_cls(x, y)
        self._metrics_layer.updae_var(self._rpn_metrics_cls.result())
        return x
    pass
#    rpn层回归评价
class RpnMetricsClsInFasterRcnn(tf.keras.metrics.Metric):
    def __init__(self, name='RpnMetricsClsInFasterRcnn', **kwargs):
        super(RpnMetricsClsInFasterRcnn, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='rpn_cls_acc', initializer='zero', dtype=tf.float32)
        pass
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
    def update_val(self, val):
        self.acc.assign(val)
        pass
    def result(self):
        return self.acc
    def reset_states(self):
        self.acc.assign(0.)
        pass
    pass

#    rpn层回归评价
class RpnMetricsRegLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 metrics_layer=None,
                 rpn_metrics_reg=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param metrics_layer: faster_rcnn的评价对象
            @param rpn_metrics_cls: rpn的分类评价对象
        '''
        super(RpnMetricsRegLayer, self).__init__(name='RpnMetricsRegLayer', **kwargs)
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        
        self._metrics_layer = metrics_layer
        self._rpn_metrics_reg = rpn_metrics_reg
        pass
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算rpn的分类损失
        self._rpn_metrics_reg(x, y)
        self._metrics_layer.updae_var(self._rpn_metrics_reg.result())
        return x
    pass
#    rpn层回归评价
class RpnMetricsRegInFasterRcnn(tf.keras.metrics.Metric):
    def __init__(self, name='RpnMetricsRegInFasterRcnn', **kwargs):
        super(RpnMetricsRegInFasterRcnn, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='rpn_reg_mae', initializer='zero', dtype=tf.float32)
        pass
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
    def update_val(self, val):
        self.acc.assign(val)
        pass
    def result(self):
        return self.acc
    def reset_states(self):
        self.acc.assign(0.)
        pass
    pass


############################################################################################################
#    fast_rcnn层评价函数层评价函数
############################################################################################################
#    rpn分类评价层
class FastRcnnMetricsClsLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 metrics_layer=None,
                 fast_rcnn_metrics_cls=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param metrics_layer: faster_rcnn的评价对象
            @param rpn_metrics_cls: rpn的分类评价对象
        '''
        super(FastRcnnMetricsClsLayer, self).__init__(name='FastRcnnMetricsClsLayer', **kwargs)
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        
        self._metrics_layer = metrics_layer
        self._fast_rcnn_metrics_cls = fast_rcnn_metrics_cls
        pass
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算rpn的分类损失
        self._fast_rcnn_metrics_cls(x, y)
        self._metrics_layer.updae_var(self._fast_rcnn_metrics_cls.result())
        return x
    pass
#    rpn层回归评价
class FastRcnnMetricsClsInFasterRcnn(tf.keras.metrics.Metric):
    def __init__(self, name='FastRcnnMetricsClsInFasterRcnn', **kwargs):
        super(FastRcnnMetricsClsInFasterRcnn, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='fast_rcnn_cls_acc', initializer='zero', dtype=tf.float32)
        pass
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
    def updae_var(self, val):
        self.acc.assign(val)
        pass
    def result(self):
        return self.acc
    def reset_states(self):
        self.acc.assign(0.)
        pass
    pass

#    rpn层回归评价
class FastRcnnMetricsRegLayer(tf.keras.layers.Layer):
    def __init__(self,
                 train_ycrt_queue=ProposalsCrtBatchQueue.default(), 
                 untrain_ycrt_queue=ProposalsCrtBatchQueue.default(),
                 metrics_layer=None,
                 fast_rcnn_metrics_reg=None,
                 **kwargs
                 ):
        '''
            @param train_ycrt_queue: 暂存训练时的y值
            @param untrain_ycrt_queue: 暂存预测时的y值
            @param metrics_layer: faster_rcnn的评价对象
            @param rpn_metrics_cls: rpn的分类评价对象
        '''
        super(FastRcnnMetricsRegLayer, self).__init__(name='FastRcnnMetricsRegLayer', **kwargs)
        
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        
        self._metrics_layer = metrics_layer
        self._fast_rcnn_metrics_reg = fast_rcnn_metrics_reg
        pass
    def call(self, x, training=None, **kwargs):
        #    取y值
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        
        #    计算rpn的分类损失
        self._fast_rcnn_metrics_reg(x, y)
        self._metrics_layer.updae_var(self._fast_rcnn_metrics_reg.result())
        return x
    pass
#    rpn层回归评价
class FastRcnnMetricsRegInFasterRcnn(tf.keras.metrics.Metric):
    def __init__(self, name='FastRcnnMetricsRegInFasterRcnn', **kwargs):
        super(FastRcnnMetricsRegInFasterRcnn, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='fast_rcnn_reg_mae', initializer='zero', dtype=tf.float32)
        pass
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
    def updae_var(self, val):
        self.acc.assign(val)
        pass
    def result(self):
        return self.acc
    def reset_states(self):
        self.acc.assign(0.)
        pass
    pass