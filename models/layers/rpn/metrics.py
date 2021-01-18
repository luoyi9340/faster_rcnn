# -*- coding: utf-8 -*-  
'''
rpn的相关评价函数

@author: luoyi
Created on 2021年1月5日
'''
import tensorflow as tf

import utils.logger_factory as logf
from models.layers.rpn.preprocess import takeout_sample


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
    def tp_tn_fp_tf_p_n(self, ymaps_cls_p, fmaps_cls_p, ymaps_cls_n, fmaps_cls_n):
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
        #    计算正负样本数量
        P = tf.math.count_nonzero(ymaps_cls_p, dtype=tf.float32)          #    有1个非0点就有1个正样本
        N = tf.math.count_nonzero(ymaps_cls_n, dtype=tf.float32)          #    有1个非0点就有1个负样本
        #    正负样本的预测结果
        prob_cls_p = tf.gather_nd(fmaps_cls_p, tf.where(fmaps_cls_p > 0))         #    过滤掉0值
        prob_cls_n = tf.gather_nd(fmaps_cls_n, tf.where(fmaps_cls_n > 0))
        ones_like_cls_p, zeros_like_cls_p = tf.ones_like(prob_cls_p), tf.zeros_like(prob_cls_p)
        ones_like_cls_n, zeros_like_cls_n = tf.ones_like(prob_cls_n), tf.zeros_like(prob_cls_n)
        TP = tf.math.count_nonzero(tf.where(prob_cls_p > 0.5, ones_like_cls_p, zeros_like_cls_p), dtype=tf.float32)       #    TP = 正样本的前景概率 > 0.5
        FP = tf.math.count_nonzero(tf.where(prob_cls_n <= 0.5, ones_like_cls_n, zeros_like_cls_n), dtype=tf.float32)       #    FP = 负样本的前景概率 > 0.5
        TN = tf.math.count_nonzero(tf.where(prob_cls_n > 0.5, ones_like_cls_n, zeros_like_cls_n), dtype=tf.float32)       #    TN = 负样本的背景概率 > 0.5
        FN = tf.math.count_nonzero(tf.where(prob_cls_p <= 0.5, ones_like_cls_p, zeros_like_cls_p), dtype=tf.float32)       #    FN = 正样本的背景概率 > 0.5
        
        return (TP, TN, FP, FN, P, N)
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
        (fmaps_cls_p, fmaps_cls_n, _), (ymaps_cls_p, ymaps_cls_n, _) = takeout_sample(y_true, y_pred)
        (tp, tn, fp, fn, p, n) = self.tp_tn_fp_tf_p_n(ymaps_cls_p, fmaps_cls_p, ymaps_cls_n, fmaps_cls_n)
        
        total = tf.math.add(p, n)
        t = tf.math.add(tp, tn)
        acc = tf.math.divide(t, total)
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
    def mean_abs_error(self, ymaps_reg_p, fmaps_reg_p, ymaps_cls_p):
        '''平均绝对误差
            mae = 1/N * ∑(|y - p|)
            @param ymaps_reg_p: 标签的t[*]
            @param fmaps_reg_p: 预测的d[*] 
            @param ymaps_cls_p: 标签的正样本前景概率
        '''
        count = tf.math.count_nonzero(ymaps_cls_p, dtype=tf.float32)          #    正样本数量
        abs_err = tf.math.abs(ymaps_reg_p - fmaps_reg_p)
        sum_abs_err = tf.math.reduce_sum(abs_err)
        mean_abs_err = sum_abs_err / count
        return mean_abs_err
    
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
        (_, _, fmaps_reg_p), (ymaps_cls_p, _, ymaps_reg_p) = takeout_sample(y_true, y_pred)
        
        mae = self.mean_abs_error(ymaps_reg_p, fmaps_reg_p, ymaps_cls_p)
        tf.print('mae:', mae, output_stream=logf.get_logger_filepath('rpn_metric'))
        self.mae.assign(mae)
        pass
    
    def result(self):
        return self.mae
    
    def reset_states(self):
        self.mae.assign(0.)
        pass
    pass
