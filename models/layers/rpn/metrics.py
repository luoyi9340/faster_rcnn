# -*- coding: utf-8 -*-  
'''
rpn的相关评价函数

@author: luoyi
Created on 2021年1月5日
'''
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, threshold=np.inf)

import utils.logger_factory as logf
from models.layers.rpn.preprocess import takeout_sample_array


#    分类评价函数
class RPNMetricCls(tf.keras.metrics.Metric):
    '''分类评价函数
        准确率 = (TP + TN) / (TP + TN + FP + FN) 
    '''
    def __init__(self, name='RPNMetricCls', **kwargs):
        super(RPNMetricCls, self).__init__(name=name, **kwargs)
        
        self.acc = self.add_weight(name='cls_acc', initializer='zero', dtype=tf.float32)
        pass
    #    计算TP, TN, FP, FN, P, N
    def tp_tn_fp_tf_p_n(self, anchors):
        ''' 
            TP：正样本前景概率 > 0.5
            TN：负样本背景概率 > 0.5
            FP：负样本前景概率 > 0.5
            FN：正样本背景概率 > 0.5
            @param ymaps_cls_p: 正样本标签(batch_size, h, w, 2, K)
            @param fmaps_cls_p: 正样本预测
            @param ymaps_cls_n: 负样本标签
            @param fmaps_cls_n: 负样本预测
            @return: (TP, TN, FP, FN, P, N)
        '''
        anchors_p = anchors[anchors[:,:,0] > 0]
        anchors_n = anchors[anchors[:,:,0] < 0]
        #    正负样本总数
        P = tf.cast(tf.math.count_nonzero(anchors_p[:,0]), dtype=tf.float32) 
        N = tf.cast(tf.math.count_nonzero(anchors_n[:,0]), dtype=tf.float32) 
        #    各种数据
        TP = tf.math.count_nonzero(anchors_p[:,1] > 0.5, dtype=tf.float32)                #    正样本 的 正样本概率>0.5 即为正样本分类正确
        FN = tf.math.count_nonzero(anchors_p[:,2] > 0.5, dtype=tf.float32)                #    正样本 的 负样本概率>0.5 即为正样本分类错误
        TN = tf.math.count_nonzero(anchors_n[:,2] > 0.5, dtype=tf.float32)                #    负样本 的 负样本概率>0.5 即为负样本分类正确
        FP = tf.math.count_nonzero(anchors_n[:,1] > 0.5, dtype=tf.float32)                #    负样本 的 正样本概率>0.5 即为负样本分类错误
        return (TP, TN, FP, FN, P, N)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
            @param y_true: [y_true, y_true...]
                                [IoU得分(正样本>0，负样本<0), idx_w, idx_h, idx_area, idx_scales, 正负样本区分(1 | -1), t[x], t[y], t[w], t[h]]
                            
            @param y_pred: [pred, pred...]
                            pred中的格式：(y_cls, y_reg)
                            y_cls.shape = (batch_size, h, w, 2, K)
                                (batch_size, h, w, 0, K)代表每个点的前景概率，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的背景概率，K的顺序为area * scales
                            y_reg.shape = (batch_size, h, w, 4, K)
                                (batch_size, h, w, 0, K)代表每个点的△x位移，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的△y位移，K的顺序为area * scales
                                (batch_size, h, w, 2, K)代表每个点的x缩放，K的顺序为area * scales
                                (batch_size, h, w, 3, K)代表每个点的y缩放，K的顺序为area * scales
        '''
        
        anchors = takeout_sample_array(y_true, y_pred)
        (tp, tn, fp, fn, p, n) = self.tp_tn_fp_tf_p_n(anchors)
        
        total = tf.math.add(p, n)
        t = tf.math.add(tp, tn)
        acc = tf.math.divide(t, total)
        acc = tf.cast(acc, dtype=tf.float32)
        tf.print('--------------------------------------------------', output_stream=logf.get_logger_filepath('rpn_metric'))
        tf.print('acc:', acc, ' t(tp+tn):', t, ' total(p+n):', total, ' tp:', tp, ' tn:', tn, ' fp:', fp, ' fn:', fn, ' p:', p, ' n:', n, output_stream=logf.get_logger_filepath('rpn_metric'))
        self.acc.assign(acc)
        pass
    def result(self):
        return self.acc
    def reset_states(self):
        self.acc.assign(0.)
        pass
    pass


#    回归评价函数
class RPNMetricReg(tf.keras.metrics.Metric):
    '''回归评价函数
        平均绝对误差 = 1/N[n] * |t[*] - d[*]|
    '''
    def __init__(self, name='RPNMetricReg', **kwargs):
        super(RPNMetricReg, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='reg_mae', initializer='zero', dtype=tf.float32)
        pass
    
    #    计算平均绝对误差
    def mean_abs_error(self, y_true, anchors):
        '''平均绝对误差
            mae = 1/N * ∑(|y - p|)
            @param ymaps_reg_p: 标签的t[*]
            @param fmaps_reg_p: 预测的d[*] 
            @param ymaps_cls_p: 标签的正样本前景概率
        '''
        #    取正样本
        anchors_p = anchors[anchors[:,:,0] > 0]
        y_true_p = y_true[y_true[:,:,5] > 0]
        dx = anchors_p[:, 3]
        dy = anchors_p[:, 4]
        dw = anchors_p[:, 5]
        dh = anchors_p[:, 6]
        tx = y_true_p[:, 6]
        ty = y_true_p[:, 7]
        tw = y_true_p[:, 8]
        th = y_true_p[:, 9]
        mae_x = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(tx, dx)))
        mae_y = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(ty, dy)))
        mae_w = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(tw, dw)))
        mae_h = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(th, dh)))
        mae = tf.math.reduce_sum([mae_x, mae_y, mae_w, mae_h])
        
        tf.print('mae:', mae, ' mae_x:', mae_x, ' mae_y:', mae_y, ' mae_w:', mae_w, ' mae_h:', mae_h, output_stream=logf.get_logger_filepath('rpn_metric'))

        return mae
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
            @param y_true: [y_true, y_true...]
                            y_true中的格式：(batch_size, h, w, 6, K)
                            y_cls.shape = (batch_size, h, w, 2, K)
                                (batch_size, h, w, 0, K)代表每个点的前景概率，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的背景概率，K的顺序为area * scales
                            y_reg.shape = (batch_size, h, w, 4, K)
                                (batch_size, h, w, 0, K)代表每个点的△x位移，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的△y位移，K的顺序为area * scales
                                (batch_size, h, w, 2, K)代表每个点的x缩放，K的顺序为area * scales
                                (batch_size, h, w, 3, K)代表每个点的y缩放，K的顺序为area * scales
                            
            @param y_pred: [pred, pred...]
                            pred中的格式：(batch_size, h, w, 6, K)
                            y_cls.shape = (batch_size, h, w, 2, K)
                                (batch_size, h, w, 0, K)代表每个点的前景概率，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的背景概率，K的顺序为area * scales
                            y_reg.shape = (batch_size, h, w, 4, K)
                                (batch_size, h, w, 0, K)代表每个点的△x位移，K的顺序为area * scales
                                (batch_size, h, w, 1, K)代表每个点的△y位移，K的顺序为area * scales
                                (batch_size, h, w, 2, K)代表每个点的x缩放，K的顺序为area * scales
                                (batch_size, h, w, 3, K)代表每个点的y缩放，K的顺序为area * scales
        '''
        anchors = takeout_sample_array(y_true, y_pred)
        
        mae = self.mean_abs_error(y_true, anchors)
        self.mae.assign(mae)
        pass
    
    def result(self):
        return self.mae
    
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass
