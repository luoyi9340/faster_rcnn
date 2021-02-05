# -*- coding: utf-8 -*-  
'''
fast_rcnn评价指标

@author: luoyi
Created on 2021年1月27日
'''
import tensorflow as tf

import utils.logger_factory as logf
from models.layers.fast_rcnn.preprocess import takeout_sample_array


#    分类评价指标
class FastRcnnMetricCls(tf.metrics.Metric):
    '''分类评价函数
        准确率 = T / Total
    '''
    def __init__(self, name='FastRcnnMetricCls', **kwargs):
        super(FastRcnnMetricCls, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='cls_acc', initializer='zero', dtype=tf.float32)
        pass
    
    #    取分类正确数量和总数
    def tn_t(self, y_true, y_pred):
        B = tf.math.count_nonzero(y_true[:,0,0] + 1)
        num = y_true.shape[1]                   #    batch_size, 每个batch_size中的proposal数
        total = B * num
        
        #    预测结果
        prob = y_pred[:,0,:]
        prob_idx = tf.argmax(prob, axis=1)
        #    真实分类
        true_idx = tf.cast(y_true[:,:,0], dtype=tf.int64)
        true_idx = tf.reshape(true_idx, shape=(total,))
        #    比较计算预测准确率
        eq = tf.equal(prob_idx, true_idx)
        T = tf.math.count_nonzero(eq)
        
        return T, total
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
            @param y_true: tensor(batch_size, num, 9)
                                1个批次代表1张图，1个num代表一张图的1个proposal
                                [
                                    [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                                    [vidx, xl,yl,xr,yr, tx,th,tw,th]
                                    ...
                                ]
            @param y_pred: tensor(batch_size*num, 5, 42)
                                (batch_size*num, 0, 42)：每个proposal预测属于每个分类的概率
                                (batch_size*num, 1, 42)：每个proposal的d[x]
                                (batch_size*num, 2, 42)：每个proposal的d[y]
                                (batch_size*num, 3, 42)：每个proposal的d[w]
                                (batch_size*num, 4, 42)：每个proposal的d[h]
        '''
        T, total = self.tn_t(y_true, y_pred)
        acc = tf.cast(T / total, dtype=tf.float32)
        self.acc.assign(acc)
        
        tf.print('--------------------------------------------------', output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        tf.print('acc:', acc, ' T:', T, ' total:', total, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        pass
    
    def result(self):
        return self.acc
        
    def reset_states(self):
        self.acc.assign(0.)
        pass
    
    pass


#    回归评价指标
class FastRcnnMetricReg(tf.metrics.Metric):
    '''回归评价指标
        MAE = 1/Nreg * ∑(m=1->Nreg)∑(i∈(x,y,w,h)) |t[m,i] - d[m,i]|
                t[m,i]：第m个样本真实t[i]
                d[m,i]：第m个样本预测d[i]
    '''
    def __init__(self, name='FastRcnnMetricReg', **kwargs):
        super(FastRcnnMetricReg, self).__init__(name, **kwargs)
        
        self.mae = self.add_weight('reg_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算平均绝对误差
    def mae_xywh(self, y_true, y_pred):
        (arrs, total, _, _) = takeout_sample_array(y_true, y_pred)
        dx = arrs[:, 1]
        dy = arrs[:, 2]
        dw = arrs[:, 3]
        dh = arrs[:, 4]
        #    取真实t[x]
        tx = tf.reshape(y_true[:,:, 5], shape=(total,))
        ty = tf.reshape(y_true[:,:, 6], shape=(total,))
        tw = tf.reshape(y_true[:,:, 7], shape=(total,))
        th = tf.reshape(y_true[:,:, 8], shape=(total,))
        mae_x = tf.abs(tx - dx)
        mae_y = tf.abs(ty - dy)
        mae_w = tf.abs(tw - dw)
        mae_h = tf.abs(th - dh)
        m = mae_x + mae_y + mae_w + mae_h
        m = tf.math.reduce_mean(m)
        
        return mae_x, mae_y, mae_w, mae_h, m
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
            @param y_true: tensor(batch_size, num, 9)
                                1个批次代表1张图，1个num代表一张图的1个proposal
                                [
                                    [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                                    [vidx, xl,yl,xr,yr, tx,th,tw,th]
                                    ...
                                ]
            @param y_pred: tensor(batch_size*num, 5, 42)
                                (batch_size*num, 0, 42)：每个proposal预测属于每个分类的概率
                                (batch_size*num, 1, 42)：每个proposal的d[x]
                                (batch_size*num, 2, 42)：每个proposal的d[y]
                                (batch_size*num, 3, 42)：每个proposal的d[w]
                                (batch_size*num, 4, 42)：每个proposal的d[h]
        '''
        mae_x, mae_y, mae_w, mae_h, m = self.mae_xywh(y_true, y_pred)
        self.mae.assign(m)
        
        tf.print('mae_x:', mae_x, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        tf.print('mae_y:', mae_y, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        tf.print('mae_w:', mae_w, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        tf.print('mae_h:', mae_h, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        tf.print('mae:', m, output_stream=logf.get_logger_filepath('fast_rcnn_metric'))
        pass
    
    def result(self):
        return self.mae
        
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass