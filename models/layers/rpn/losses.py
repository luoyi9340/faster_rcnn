# -*- coding: utf-8 -*-  
'''
rpn的相关损失

@author: luoyi
Created on 2021年1月5日
'''
import tensorflow as tf

import utils.conf as conf
import utils.math_expand as me
import utils.logger_factory as logf
from models.layers.rpn.preprocess import takeout_sample

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
        (fmaps_cls_p, fmaps_cls_n, fmaps_reg_p), (ymaps_cls_p, ymaps_cls_n, ymaps_reg_p) = takeout_sample(y_true, y_pred)
        loss = self.loss_cls(ymaps_cls_p, fmaps_cls_p, ymaps_cls_n, fmaps_cls_n) \
                        + self.__loss_lamda * self.loss_reg(ymaps_reg_p, fmaps_reg_p, ymaps_cls_p)
        return loss
    
    #    loss_cls
    def loss_cls(self, ymaps_cls_p, fmaps_cls_p, ymaps_cls_n, fmaps_cls_n):
        '''交叉熵 损失
            @param ymaps_cls_p: 正样本标签(batch_size, h, w, 2, k)
            @param fmaps_cls_p: 正样本预测
            @param ymaps_cls_n: 负样本标签
            @param fmaps_cls_n: 负样本预测
            Loss_cls = 1/N * Σ(i=-1>N) -log(p_pred[i] * p_true[i] + (1 - p_pred[i]) * (1 - p_true[i]))
                对于正样本：loss = 1/N[p] Σ(i=-1>N[p]) (-log(p_pred[i] * p_true[i]))
                对于负样本：loss = 1/N[n] Σ(i=-1>N[n]) (-log((1 - p_pred[i]) * (1 - p_true[i])))
        '''
        #    计算正样本loss
        loss_cls_p = -ymaps_cls_p * tf.math.log(fmaps_cls_p)
        loss_cls_p = tf.where(tf.math.is_nan(loss_cls_p), tf.zeros_like(loss_cls_p), loss_cls_p)        #    ymaps_cls_p中存在大量的0值，过log函数会变成nan
        # loss_cls_p = tf.reduce_mean(loss_cls_p, axis=(1,2,3,4))                                       #    大量的0值也会占用mean的计算名额，所以不能用reduce_mean
        count_p = tf.cast(tf.math.count_nonzero(ymaps_cls_p, axis=(1,2,3,4)), dtype=tf.float32)          #    每个batch_size正样本总数
        loss_cls_p_sum = tf.math.reduce_sum(loss_cls_p, axis=(1,2,3,4))
        loss_cls_p = loss_cls_p_sum / count_p
        
        #    计算负样本loss
        loss_cls_n = ymaps_cls_n * tf.math.log(fmaps_cls_n)                                             #    负样本的值本身就是-1
        loss_cls_n = tf.where(tf.math.is_nan(loss_cls_n), tf.zeros_like(loss_cls_n), loss_cls_n)        #    跟ymaps_cls_p的道理一样
        # loss_cls_n = tf.reduce_mean(loss_cls_n, axis=(1,2,3,4))                                       #    跟ymaps_cls_p的道理一样
        count_n = tf.cast(tf.math.count_nonzero(ymaps_cls_n, axis=(1,2,3,4)), dtype=tf.float32)
        loss_cls_n_sum = tf.math.reduce_sum(loss_cls_n, axis=(1,2,3,4))
        loss_cls_n = loss_cls_n_sum / count_n
        
        loss_cls = loss_cls_p + loss_cls_n
        
        tf.print('--------------------------------------------------', output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_cls_p:', loss_cls_p, ' loss_cls_p_sum:', loss_cls_p_sum, ' count_p:', count_p, output_stream=logf.get_logger_filepath('rpn_loss'))
        tf.print('loss_cls_n:', loss_cls_n, ' loss_cls_n_sum:', loss_cls_n_sum, ' count_n:', count_n, output_stream=logf.get_logger_filepath('rpn_loss'))
        return loss_cls


    #    loss_reg
    def loss_reg(self, ymaps_reg_p, fmaps_reg_p, ymaps_cls_p):
        '''smoothL1 损失
            @param ymaps_reg_p: 正样本标签(batch_size, h, w, 4, K)
            @param fmaps_reg_p: 正样本预测
            @param y_true_cls: 计算分类损失时候的正样本(batch_size, h, w, 0, K)
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
        #    计算正样本loss
        loss_reg_p = me.smootL1_tf(ymaps_reg_p - fmaps_reg_p)
        count_p = tf.cast(tf.math.count_nonzero(ymaps_cls_p, axis=(1,2,3,4)), dtype=tf.float32)
        loss_reg_p_sum = tf.math.reduce_sum(loss_reg_p, axis=(1,2,3,4))
        loss_reg_p = loss_reg_p_sum / count_p
        
        tf.print('loss_reg:', loss_reg_p, ' loss_reg_p_sum:', loss_reg_p_sum, ' count_p:', count_p, output_stream=logf.get_logger_filepath('rpn_loss'))
        return loss_reg_p
    pass
