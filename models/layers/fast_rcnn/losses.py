# -*- coding: utf-8 -*-  
'''
fast_rcnn的损失
    loss = loss_cls + λ * loss_reg
    loss_cls = 1/Ncls * ∑(m=1->Ncls)∑(i=1->42) (p[m,i] * log(d[m,i]))
                    p[m,i]为第m个建议框的真实分类，第i个为1，其他为0
                    d[m,i]为第m个建议框的预测分类，所以这里只要算真实分类的预测即可。其他的算出来外面也要*0
    loss_reg = 1/Nreg * ∑(m=1->Nreg)∑(t∈(x,y,w,h)) smootL1(t[m,*] - d[m,*])
                    t[m,*]为第m个样本的真实偏移比/缩放比
                    d[m,*]为第m个样本的预测偏移比/缩放比

@author: luoyi
Created on 2021年1月27日
'''
import tensorflow as tf

import utils.conf as conf
import utils.logger_factory as logf
import utils.math_expand as me
from models.layers.fast_rcnn.preprocess import takeout_sample_array

log = logf.get_logger('fast_rcnn_loss')


#    fast_rcnn的损失函数
class FastRcnnLoss(tf.losses.Loss):
    def __init__(self, loss_lamda=conf.FAST_RCNN.get_loss_lamda(), **kwargs):
        super(FastRcnnLoss, self).__init__(**kwargs)
        
        self.__loss_lamda = loss_lamda
        pass
    
    #    损失函数定义
    def call(self, y_true, y_pred):
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
        tf.print('--------------------------------------------------', output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        arrs = takeout_sample_array(y_true, y_pred)
        loss = self.loss_cls(arrs) + self.__loss_lamda * self.loss_reg(y_true, arrs)
        
        tf.print('loss:', loss, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        return loss

    
    #    分类损失
    def loss_cls(self, arrs):
        '''分类损失
            loss_cls = 1/Ncls * ∑(m=1->Ncls)∑(i=1->C) -1 * y_true[m,i] * log(y_pred[m,i])
                            y_true[m,i]：第m个样本的第i个分类真实概率。只有1个为1，其他为0
                            y_pred[m,i]：第m个样本的第i个分类的预测概率
                            
            @param arrs: tensor(batch_size, num, 5) y_true中的vidx对应y_pred中的预测向量
                            [分类概率，dx, dy, dw, dh]
            @return: loss_cls tensor(batch_size, 1)
        '''
        #    1个42*5代表y_true中1条记录
        probs = arrs[:,0]                 #    y_true中的vidx在y_pred中对应的预测概率
        #    计算交叉熵
        loss_cls = -1 * tf.math.log(probs)         
        
        tf.print('loss_cls:', loss_cls, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        return loss_cls
    
    #    回归损失
    def loss_reg(self, y_true, arrs):
        '''回归损失
            loss_reg = 1/Nreg * ∑(m=1->Nreg)∑(i∈(x,y,w,h)) smootL1(t[m,i] - d[m,i])
                            t[m,i]：第m个样本的真实t[i]值
                            d[m,i]：第m个样本的预测d[i]值
        
            @param y_true: tensor(batch_size, num, 9)
                                1个批次代表1张图，1个num代表一张图的1个proposal
                                [
                                    [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                                    [vidx, xl,yl,xr,yr, tx,th,tw,th]
                                    ...
                                ]
            @param arrs: tensor(batch_size, num, 5) y_true中的vidx对应y_pred中的预测向量
                            [分类概率，dx, dy, dw, dh]
            @return: loss_reg tensor(batch_size, 1)
        '''
        B, num = y_true.shape[0], y_true.shape[1]
        total = B * num
        #    取预测d[*]
        dx = arrs[:, 1]
        dy = arrs[:, 2]
        dw = arrs[:, 3]
        dh = arrs[:, 4]
        #    取真实t[x]
        tx = tf.reshape(y_true[:,:, 5], shape=(total,))
        ty = tf.reshape(y_true[:,:, 6], shape=(total,))
        tw = tf.reshape(y_true[:,:, 7], shape=(total,))
        th = tf.reshape(y_true[:,:, 8], shape=(total,))
        #    计算smootL1
        loss_x = me.smootL1_tf(tx - dx)
        loss_y = me.smootL1_tf(ty - dy)
        loss_w = me.smootL1_tf(tw - dw)
        loss_h = me.smootL1_tf(th - dh)
        loss_reg = tf.math.reduce_sum([loss_x, loss_y, loss_w, loss_h], axis=0)  
        
        tf.print('loss_x:', loss_x, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        tf.print('loss_y:', loss_y, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        tf.print('loss_w:', loss_w, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        tf.print('loss_h:', loss_h, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        tf.print('loss_reg:', loss_reg, output_stream=logf.get_logger_filepath('fast_rcnn_loss'))
        return loss_reg
    pass