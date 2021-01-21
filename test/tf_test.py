# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import numpy as np

import utils.math_expand as me
from models.layers.rpn.preprocess import preprocess_like_array, takeout_sample_array  

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  
# t = tf.ones(shape=[2, 2,2, 3])
# print(t)
# t = tf.pad(t, [[0,0], [1,1],[1,1], [0,0]])
# print(t)


#    假设40*40的原图，缩放比例10，特征图4*4
roi_areas = np.array([10, 20])
roi_scales = np.array([1, 2])
#    [IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x(label左上点), y(label左上点), w, h]
y_true = [
            [0.9, 5,5,10,10, 0,0,0,0, 1, 0,0,10,10],
            [0.9, 15,15,40,10, 1,1,1,1, -1, 10,35,40,10],
            [0.9, 25,25,20,20, 2,2,1,0, 1, 15,15,20,20],
            [0.9, 35,35,20,5, 3,3,0,1, -1, 37.5,45,20,5]
        ]
y_true = np.array(y_true)
y = preprocess_like_array(y_true)
y = np.expand_dims(y, axis=0)
y = np.repeat(y, 5, axis=0)
#    模拟标签数据[batch_size=4, num=4, 10]
y_true = tf.convert_to_tensor(y, dtype=tf.float32)
#    模拟特征图
fmaps_cls = tf.random.uniform(shape=(5, 4, 4, 2, 4), dtype=tf.float32)
fmaps_cls = tf.nn.softmax(fmaps_cls, axis=3)
fmaps_reg = tf.random.uniform(shape=(5, 4, 4, 4, 4), dtype=tf.float32)
fmaps = tf.concat([fmaps_cls, fmaps_reg], axis=3)
# print(fmaps)
anchors = takeout_sample_array(y_true, fmaps)
print(anchors)


#    计算分类loss
#    取需要计算的各自概率（正样本取[:,1]，负样本[:,2]）
loss_d = tf.where(anchors[:,:,0] > 0, anchors[:,:,1], anchors[:,:,2])
loss_cls = -1 * tf.math.log(loss_d)
loss_cls = tf.reduce_mean(loss_cls, axis=1)
# print(loss_cls)

#    计算回归loss
zero_tmp = tf.zeros_like(anchors)
one_tmp = tf.ones_like(anchors)
idx_p = y_true[:,:,0] > 0
count_p = tf.cast(tf.math.count_nonzero(idx_p, axis=1), dtype=tf.float32)                  #    每个batch的正样本数
dx = tf.where(idx_p, anchors[:,:, 3], zero_tmp[:,:, 3])
dy = tf.where(idx_p, anchors[:,:, 4], zero_tmp[:,:, 4])
dw = tf.where(idx_p, anchors[:,:, 5], zero_tmp[:,:, 5])
dh = tf.where(idx_p, anchors[:,:, 6], zero_tmp[:,:, 6])
tx = tf.where(idx_p, y_true[:,:, 6], zero_tmp[:,:, 3])
ty = tf.where(idx_p, y_true[:,:, 7], zero_tmp[:,:, 4])
tw = tf.where(idx_p, y_true[:,:, 8], zero_tmp[:,:, 5])
th = tf.where(idx_p, y_true[:,:, 9], zero_tmp[:,:, 6])
loss_x = me.smootL1_tf(tf.math.subtract(tx, dx))
loss_x = tf.divide(tf.reduce_sum(loss_x, axis=1), count_p)
loss_y = me.smootL1_tf(tf.math.subtract(ty, dy))
loss_y = tf.divide(tf.reduce_sum(loss_y, axis=1), count_p)
loss_w = me.smootL1_tf(tf.math.subtract(tw, dw))
loss_w = tf.divide(tf.reduce_sum(loss_w, axis=1), count_p)
loss_h = me.smootL1_tf(tf.math.subtract(th, dh))
loss_h = tf.divide(tf.reduce_sum(loss_h, axis=1), count_p)
# print('loss_x:', loss_x)
# print('loss_y:', loss_y)
# print('loss_w:', loss_w)
# print('loss_h:', loss_h)
loss_reg = tf.reduce_sum([loss_x, loss_y, loss_w, loss_h], axis=0)
# print('loss_reg:', loss_reg)


#    计算评价标准
#    计算分类评价标准
#    正负样本
anchors_p = anchors[anchors[:,:,0] > 0]
anchors_n = anchors[anchors[:,:,0] < 0]
#    正负样本总数
P = tf.cast(tf.math.count_nonzero(anchors_p[:,0]), dtype=tf.float32) 
N = tf.cast(tf.math.count_nonzero(anchors_n[:,0]), dtype=tf.float32) 
print(P)
print(N)
#    各种数据
TP = tf.math.count_nonzero(anchors_p[:,1] > 0.5)                #    正样本 的 正样本概率>0.5 即为正样本分类正确
FN = tf.math.count_nonzero(anchors_p[:,2] > 0.5)                #    正样本 的 负样本概率>0.5 即为正样本分类错误
TN = tf.math.count_nonzero(anchors_n[:,2] > 0.5)                #    负样本 的 负样本概率>0.5 即为负样本分类正确
FP = tf.math.count_nonzero(anchors_n[:,1] > 0.5)                #    负样本 的 正样本概率>0.5 即为负样本分类错误
print(TP, FN, TN, FP, P, N)

#    回归评价函数
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
print(mae)


