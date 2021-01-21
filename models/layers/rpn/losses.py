# -*- coding: utf-8 -*-  
'''
损失函数

@author: luoyi
Created on 2021年1月19日
'''
import tensorflow as tf

import utils.conf as conf
import utils.math_expand as me
import utils.logger_factory as logf
from models.layers.rpn.preprocess import takeout_sample_array


#    RPN网络的损失
class RPNLoss(tf.keras.losses.Loss):
    '''RPN网络 损失函数
        loss = Loss_cls + λ * Loss_reg
            Loss_cls = 1/N * Σ(i=-1>N) -log(p_pred[i] * p_true[i] + (1 - p_pred[i]) * (1 - p_true[i]))
                对于正样本：loss = 1/N[p] Σ(i=-1>N[p]) (-log(p_pred[i] * p_true[i]))
                对于负样本：loss = 1/N[n] Σ(i=-1>N[n]) (-log((1 - p_pred[i]) * (1 - p_true[i])))
            Loss_reg = ∑(j=[x,y,w,h]) 1/N[p] * ∑(i=1->N[p]) smoothL1(t[j] - d[j])
                     = ∑(i=1->N[p]) smoothL1(t[x] - d[x])
                         +
                        ∑(i=1->N[p]) smoothL1(t[y] - d[y])
                         +
                        ∑(i=1->N[p]) smoothL1(t[w] - d[w])
                         +
                        ∑(i=1->N[p]) smoothL1(t[h] - d[h])
    '''
    def __init__(self, loss_lamda=conf.RPN.get_loss_lamda(), **kwarg):
        '''
            @param loss_lamda: loss损失中Loss_reg权重占比，>0的任何数
        '''
        super(RPNLoss, self).__init__(**kwarg)
        
        self.__loss_lamda = loss_lamda
        pass
    
    
    #    损失函数
    def call(self, y_true, y_pred):
        '''loss = Loss_cls + λ * Loss_reg    （详见类注释）
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
        tf.print('--------------------------------------------------', output_stream=logf.get_logger_filepath('rpn_loss'))
        anchors = takeout_sample_array(y_true, y_pred)
        loss = self.loss_cls(anchors) + self.__loss_lamda * self.loss_reg(anchors, y_true)
        #    无脑sum
        loss = tf.reduce_sum(loss)
        tf.print('loss:', loss, output_stream=logf.get_logger_filepath('rpn_loss'))
        return loss
    
    #    loss_cls
    def loss_cls(self, anchors):
        '''交叉熵 损失
            @param fmaps_p: Tensor (num, 6) 每条正样本的[正样本概率，负样本概率，d[x], d[y], d[w], d[h]]
            @param fmaps_n: Tensor (num, 6) 每条正样本的[正样本概率，负样本概率，d[x], d[y], d[w], d[h]]
            Loss_cls = 1/N * Σ(i=-1>N) -log(p_pred[i] * p_true[i] + (1 - p_pred[i]) * (1 - p_true[i]))
                对于正样本：loss = 1/N[p] Σ(i=-1>N[p]) (-log(p_pred[i] * p_true[i]))
                对于负样本：loss = 1/N[n] Σ(i=-1>N[n]) (-log((1 - p_pred[i]) * (1 - p_true[i])))
        '''
        #    取需要计算的各自概率（正样本取[:,1]，负样本[:,2]）
        loss_d = tf.where(anchors[:,:,0] > 0, anchors[:,:,1], anchors[:,:,2])
        loss_cls = -1 * tf.math.log(loss_d)
        loss_cls = tf.reduce_mean(loss_cls, axis=1)
        
        tf.print('loss_cls:', loss_cls, output_stream=logf.get_logger_filepath('rpn_loss'))
        return loss_cls

    #    loss_reg
    def loss_reg(self, anchors, y_true):
        '''smoothL1 损失
            @param fmaps_p: Tensor (num, 6) 每条正样本的[正样本概率，负样本概率，d[x], d[y], d[w], d[h]]
            @param y_true: Tensor (batch_size) [IoU得分(正样本>0，负样本<0), idx_w, idx_h, idx_area, idx_scales, 正负样本区分(1 | -1), t[x], t[y], t[w], t[h]]
            loss_reg = ∑(j=[x,y,w,h]) 1/N[p] * ∑(i=1->N[p]) smoothL1(t[j] - d[j])
                     = ∑(i=1->N[p]) smoothL1(t[x] - d[x])
                         +
                        ∑(i=1->N[p]) smoothL1(t[y] - d[y])
                         +
                        ∑(i=1->N[p]) smoothL1(t[w] - d[w])
                         +
                        ∑(i=1->N[p]) smoothL1(t[h] - d[h])
                目标是让模型产出的候选框与anchor的偏移/缩放量 尽可能接近 标记的候选框与anchor的偏移/缩放量
                其中：t[*]为标记数据计算出的偏移比/缩放比
                         t[x] = (G[x] - P[x]) * P[w]        G[x] = t[x]/P[w] + P[x]（t[x]/P[w]为真实偏移量）
                         t[y] = (G[y] - P[y]) * P[h]        G[y] = t[y]/P[h] + P[y]（t[y]/P[h]为真实偏移量）
                         t[w] = log(G[w] / P[w])            G[w] = exp(t[w]) * P[w]（exp(t[w])为真实缩放）
                         t[h] = log(G[h] / P[h])            G[h] = exp(t[h]) * P[h]（exp(t[h])为真实缩放）
                     d[*]为预测给出的偏移比/缩放比
                         d[x] = (G_[x] - P[x]) * P[w]       G_[x] = d[x]/P[w] + P[x]（d[x]/P[w]为真实偏移量）
                         d[y] = (G_[y] - P[y]) * P[h]       G_[y] = d[y]/P[h] + P[y]（d[y]/P[h]为真实偏移量）
                         d[w] = log(G_[w] / P[w])           G_[w] = exp(d[w]) * P[w]（exp(d[w])为真实缩放）
                         d[h] = log(G_[h] / P[h])           G_[h] = exp(d[h]) * P[h]（exp(d[h])为真实缩放）
                     G[x]为标记框中心点x
                     G[y]为标记框中心点y
                     G[w]为标记框宽度
                     G[h]为标记框高度
                     P[x]为anchor框中心点x
                     P[y]为anchor框中心点y
                     P[w]为anchor框宽度
                     P[h]为anchor框高度
                     G_[x]为预测框中心点x
                     G_[y]为预测框中心点y
                     G_[w]为预测框宽度
                     G_[h]为预测框高度
        '''
        zero_tmp = tf.zeros_like(anchors)
        idx_p = y_true[:,:,0] > 0
        #    每个batch的正样本数，其实不用取，肯定==正负样本数之和
        count_p = tf.cast(tf.math.count_nonzero(idx_p, axis=1), dtype=tf.float32)                  
        #    取d[*], t[*]
        dx = tf.where(idx_p, anchors[:,:, 3], zero_tmp[:,:, 3])
        dy = tf.where(idx_p, anchors[:,:, 4], zero_tmp[:,:, 4])
        dw = tf.where(idx_p, anchors[:,:, 5], zero_tmp[:,:, 5])
        dh = tf.where(idx_p, anchors[:,:, 6], zero_tmp[:,:, 6])
        tx = tf.where(idx_p, y_true[:,:, 6], zero_tmp[:,:, 3])
        ty = tf.where(idx_p, y_true[:,:, 7], zero_tmp[:,:, 4])
        tw = tf.where(idx_p, y_true[:,:, 8], zero_tmp[:,:, 5])
        th = tf.where(idx_p, y_true[:,:, 9], zero_tmp[:,:, 6])
        #    计算1/Nreg * smootL1(t[*] - d[*])
        loss_x = me.smootL1_tf(tf.math.subtract(tx, dx))
        loss_x = tf.divide(tf.reduce_sum(loss_x, axis=1), count_p)
        loss_y = me.smootL1_tf(tf.math.subtract(ty, dy))
        loss_y = tf.divide(tf.reduce_sum(loss_y, axis=1), count_p)
        loss_w = me.smootL1_tf(tf.math.subtract(tw, dw))
        loss_w = tf.divide(tf.reduce_sum(loss_w, axis=1), count_p)
        loss_h = me.smootL1_tf(tf.math.subtract(th, dh))
        loss_h = tf.divide(tf.reduce_sum(loss_h, axis=1), count_p)
        #    batch求和
        loss_reg = tf.reduce_sum([loss_x, loss_y, loss_w, loss_h], axis=0)
        
        tf.print('loss_x:', loss_x, output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_y:', loss_y, output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_w:', loss_w, output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_h:', loss_h, output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_reg:', loss_reg, output_stream=logf.get_logger_filepath('rpn_loss'))
        return loss_reg
    pass



