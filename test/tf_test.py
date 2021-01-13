# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import tensorflow.python.framework.ops as ops
import numpy as np
import math
import sys

import utils.math_expand as em
import utils.alphabet as al
import models.layers.roi_pooling as roi_pooling
from tensorflow.python.ops.gen_sdca_ops import sdca_fprint


print(tf.version.VERSION)
# x = tf.convert_to_tensor([
#         [[1,2,3,1,2,3]],
#         [[1,2,3,1,2,3]]
#     ], dtype=tf.float32)
# x = tf.reshape(x, shape=(2, 1, 2, 3))
# print(x)
# y = tf.nn.softmax(x, axis=-2)
# print(y.numpy())

#    tensor是否支持dict
# a = {'a':'a', 'b':[[1,2,3], 1,2,3], 'c':[1,2,3]}
# a = tf.convert_to_tensor(a)
# print(a)


#    json中的数据拉直成(*, 14维数组)
# arr = [
#         [0.7, [1, 2, 10, 10, 0, 0, 0, 0], ['A', 1, 1, 10, 10]],
#         [-0.7, [1, 2, 10, 10, 0, 0, 1, 1], ['B', 1, 1, 10, 10]],
#         [0.7, [1, 2, 10, 10, 0, 0, 2, 2], ['C', 1, 1, 10, 10]],
#         [-0.7, [1, 2, 10, 10, 0, 0, 3, 3], ['D', 1, 1, 10, 10]],
#         [0.7, [1, 2, 10, 10, 0, 0, 4, 4], ['E', 1, 1, 10, 10]],
#         [-0.7, [1, 2, 10, 10, 0, 0, 5, 5], ['F', 1, 1, 10, 10]]
#     ]
# #    不规则张量测试
# arr = [[a[0]] + a[1] + [al.category_index(a[2][0])] + a[2][1:] for a in arr]
# t = tf.convert_to_tensor(arr, dtype=tf.float32)
# t = t[(t[:,0] > 0)]
# print(t)




#    loss计算模拟程序
# fmaps = tf.random.uniform(shape=(2, 1, 1, 2, 4), minval=-1, maxval=1)     #    特征图简单表示为batch_size=2, 只有1个点, K=4的模式
# fmaps = tf.nn.softmax(fmaps, axis=3)
# print(fmaps)
# y_true = tf.where(fmaps > 0.5, tf.ones_like(fmaps), tf.zeros_like(fmaps))
# print(y_true)
# y_cross_entropy = -(y_true * tf.math.log(fmaps))
# print(y_cross_entropy)
# y_cross_entropy = tf.reduce_mean(y_cross_entropy, axis=(1,2,3,4))
# print(y_cross_entropy.shape)
# print(y_cross_entropy)

# fmaps = tf.ones(shape=(2, 1, 1, 4, 4), dtype=tf.float32) * 2     #    特征图简单表示为batch_size=2, 只有1个点, K=4的模式
# y_true = tf.ones(shape=(2, 1, 1, 4, 4), dtype=tf.float32)
# y = y_true - fmaps
# y_smootL1 = em.smootL1_tf(y)
# print(y_smootL1)
# y_smootL1 = tf.reduce_mean(y_smootL1, axis=(1,2,3,4))
# print(y_smootL1)




#    metrics计算模拟程序
#    计算准确率: (TP + TN) / (TP + TN + FP + FN)
# fmaps = tf.random.uniform(shape=(2, 1, 1, 2, 4), minval=-1, maxval=1, dtype=tf.float32)     #    特征图简单表示为batch_size=2, 只有1个点, K=4的模式
# fmaps = tf.nn.softmax(fmaps, axis=3)
# print(fmaps)
# y_true = tf.convert_to_tensor([[[[[1, 0, 0, 0],
#                                   [0, 0, 1, 1]]]],
#                                   [[[[0, 0, 1, 0],
#                                   [1, 0, 0, 1]]]]], dtype=tf.float32)
# print(y_true)
# y_true_positives = y_true[:,:,:,0,:]
# y_true_negative = y_true[:,:,:,1,:]
# fmaps_pisitives = fmaps[:,:,:,0,:]                  #    预测正样本的前景概率
# fmaps_negative = fmaps[:,:,:,1,:]                   #    预测负样本的背景概率
# P = tf.math.count_nonzero(y_true_positives)         #    正样本数量
# N = tf.math.count_nonzero(y_true_negative)          #    负样本数量
# prob_pisitives = fmaps_pisitives * y_true_positives
# prob_pisitives = tf.gather_nd(prob_pisitives, tf.where(prob_pisitives > 0))         #    所有正样本的预测结果
# prob_negative = fmaps_negative * y_true_negative
# prob_negative = tf.gather_nd(prob_negative, tf.where(prob_negative > 0))            #    所有负样本的预测结果
# print(prob_pisitives, prob_negative)
# TP = tf.math.count_nonzero(tf.where(prob_pisitives > 0.5, tf.ones_like(prob_pisitives), tf.zeros_like(prob_pisitives)))
# FN = tf.math.count_nonzero(tf.where(prob_pisitives <= 0.5, tf.ones_like(prob_pisitives), tf.zeros_like(prob_pisitives)))
# TN = tf.math.count_nonzero(tf.where(prob_negative > 0.5, tf.ones_like(prob_negative), tf.zeros_like(prob_negative)))
# FP = tf.math.count_nonzero(tf.where(prob_negative <= 0.5, tf.ones_like(prob_negative), tf.zeros_like(prob_negative)))
# print((TP + TN) / (FP + FN + TP + TN))
#    计算绝对误差

#    模拟计算loss，假设特征图是2*2的，batch_size=2
#    全0的点既不是正样本也不是负样本
#    正样本 > 0, 负样本 < 0
# fmaps_cls = tf.random.uniform(shape=(2, 2, 2, 2, 4), minval=-1, maxval=1, dtype=tf.float32)     
# fmaps_cls = tf.nn.softmax(fmaps_cls, axis=3)
# fmaps_reg = tf.random.uniform(shape=(2, 2, 2, 4, 4), minval=1, maxval=50, dtype=tf.float32)     
# fmaps = tf.concat([fmaps_cls, fmaps_reg], axis=3)
# [fmaps_cls, fmaps_reg] = tf.split(fmaps, [2, 4], axis=3)
# 
# y_true = np.random.uniform(size=(2, 2,2, 6, 4), low=1, high=50)
# zero = np.zeros(shape=(6, 4))
# y_true[0, 0,1] = zero
# y_true[0, 1,1] = zero
# y_true[1, 0,0] = zero
# y_true[1, 1,0] = zero
# y_true[0, 0,0, 0] = np.array([1,0,0,1])
# y_true[0, 0,0, 1] = np.array([0,1,1,0])
# y_true[0, 1,0, 0] = np.array([0,1,1,0])
# y_true[0, 1,0, 1] = np.array([1,0,0,1])
# y_true[1, 0,1, 0] = np.array([0,1,0,1])
# y_true[1, 0,1, 1] = np.array([1,0,1,0])
# y_true[1, 1,1, 0] = np.array([0,0,1,0])
# y_true[1, 1,1, 1] = np.array([0,0,1,0])
# y_true[0, 0,0] = -y_true[0, 0,0]
# y_true[1, 1,1] = -y_true[1, 1,1]
# y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
# [y_true_cls, y_true_reg] = tf.split(y_true, [2, 4], axis=3)
# y_true_reg = tf.where(y_true_reg < 0, tf.zeros_like(y_true_reg), y_true_reg)

#    从y_true不为0的点中取
#    从>0的点中取正样本，<0的点中取负样本
# zeros_tmp_cls = tf.zeros_like(fmaps_cls)
# zeros_tmp_reg = tf.zeros_like(fmaps_reg)
# fmaps_cls_p = tf.where(y_true_cls > 0, fmaps_cls, zeros_tmp_cls)
# fmaps_cls_n = tf.where(y_true_cls < 0, fmaps_cls, zeros_tmp_cls)
# ymaps_cls_p = tf.where(y_true_cls > 0, y_true_cls, zeros_tmp_cls)
# ymaps_cls_n = tf.where(y_true_cls < 0, y_true_cls, zeros_tmp_cls)
# fmaps_reg_p = tf.where(y_true_reg > 0, fmaps_reg, zeros_tmp_reg)        #    回归只需要正样本
# ymaps_reg_p = tf.where(y_true_reg > 0, y_true_reg, zeros_tmp_reg)
#    分类的正负样本
# print(y_true_cls)
# print(ymaps_cls_p)
# print((fmaps_cls_p))
# print(ymaps_cls_n)
# print(fmaps_cls_n)
#    回归的正样本
# print(y_true_reg)
# print(ymaps_reg_p)
# print(fmaps_reg_p)


#    模拟计算loss
#    计算分类loss
# loss_cls_p = -ymaps_cls_p * tf.math.log(fmaps_cls_p)
# loss_cls_p = tf.where(tf.math.is_nan(loss_cls_p), tf.zeros_like(loss_cls_p), loss_cls_p)        #    ymaps_cls_p中存在大量的0值，过log函数会变成nan
# print(loss_cls_p)
# # loss_cls_p = tf.reduce_mean(loss_cls_p, axis=(1,2,3,4))                                       #    大量的0值也会占用mean的计算名额，所以不能用reduce_mean
# count_p = tf.cast(tf.math.count_nonzero(loss_cls_p, axis=(1,2,3,4)), dtype=tf.float32)          #    每个batch_size正样本总数
# loss_cls_p = tf.math.reduce_sum(loss_cls_p, axis=(1,2,3,4)) / count_p
# loss_cls_n = ymaps_cls_n * tf.math.log(fmaps_cls_n)                                             #    负样本的值本身就是-1
# loss_cls_n = tf.where(tf.math.is_nan(loss_cls_n), tf.zeros_like(loss_cls_n), loss_cls_n)        #    跟ymaps_cls_p的道理一样
# print(loss_cls_n)
# # loss_cls_n = tf.reduce_mean(loss_cls_n, axis=(1,2,3,4))                                       #    跟ymaps_cls_p的道理一样
# count_n = tf.cast(tf.math.count_nonzero(loss_cls_n, axis=(1,2,3,4)), dtype=tf.float32)
# loss_cls_n = tf.math.reduce_sum(loss_cls_n, axis=(1,2,3,4)) / count_n
# print(loss_cls_p, loss_cls_n)

#    计算回归loss
# loss_reg_p = em.smootL1_tf(ymaps_reg_p - fmaps_reg_p)
# print(loss_reg_p)
# count_p = tf.cast(tf.math.count_nonzero(loss_reg_p, axis=(1,2,3,4)), dtype=tf.float32)          #    ∑(x,y,h,w)∑(正样本)的数量
# loss_reg_p = tf.math.reduce_sum(loss_reg_p, axis=(1,2,3,4)) / count_p
# print(loss_reg_p)


#    模拟计算metric
#    计算分类metric（准确率 = TP + TN / TP + TN + FP + FN）
# P = tf.math.count_nonzero(ymaps_cls_p)          #    有1个非0点就有1个正样本
# N = tf.math.count_nonzero(ymaps_cls_n)          #    有1个非0点就有1个负样本
# #    正负样本的预测结果
# prob_cls_p = tf.gather_nd(fmaps_cls_p, tf.where(fmaps_cls_p > 0))         #    过滤掉0值
# print(prob_cls_p)
# prob_cls_n = tf.gather_nd(fmaps_cls_n, tf.where(fmaps_cls_n > 0))
# print(prob_cls_n)
# ones_like_cls_p, zeros_like_cls_p = tf.ones_like(prob_cls_p), tf.zeros_like(prob_cls_p)
# ones_like_cls_n, zeros_like_cls_n = tf.ones_like(prob_cls_n), tf.zeros_like(prob_cls_n)
# TP = tf.math.count_nonzero(tf.where(prob_cls_p > 0.5, ones_like_cls_p, zeros_like_cls_p))       #    TP = 正样本的前景概率 > 0.5
# FP = tf.math.count_nonzero(tf.where(prob_cls_n < 0.5, ones_like_cls_n, zeros_like_cls_n))       #    FP = 负样本的前景概率 > 0.5
# TN = tf.math.count_nonzero(tf.where(prob_cls_n > 0.5, ones_like_cls_n, zeros_like_cls_n))       #    TN = 负样本的背景概率 > 0.5
# FN = tf.math.count_nonzero(tf.where(prob_cls_p < 0.5, ones_like_cls_p, zeros_like_cls_p))       #    FN = 正样本的背景概率 > 0.5
# print(P)
# print(N)
# print(TP)
# print(FP)
# print(TN)
# print(FN)

#    计算回归metric（平均绝对误差）
# count = tf.math.count_nonzero(ymaps_cls_p, dtype=tf.float32)          #    正样本数量
# abs_ = tf.math.abs(ymaps_reg_p - fmaps_reg_p)
# print(tf.math.reduce_sum(abs_), count)
# print(tf.math.reduce_sum(abs_) / count)


#    初始化控制台输出格式
def console_handler(log_fmt="%(asctime)s-%(name)s-%(levelname)s-%(message)s", log_level=tf._logging.INFO):
    console_handler = tf._logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    #    root log输出格式
    console_fmt = tf._logging.Formatter(log_fmt)
    console_handler.setFormatter(console_fmt)
    return console_handler


#    日志
# log = tf._logging.getLogger('aaaa')
# #    打印到文件
# log_level = tf._logging.INFO
# log.setLevel(log_level)
# log_path = '/Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/logs/tf_test.log'
#     
# log_handler = tf._logging.FileHandler(log_path, encoding='utf-8')
# log_handler.setLevel(log_level)
#     
# log_fmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'
# fmt = tf._logging.Formatter(log_fmt)
# log_handler.setFormatter(fmt)
# log.addHandler(log_handler)
# 
# #    打印到控制台
# log.addHandler(console_handler(log_fmt=log_fmt, log_level=log_level))
# 
# 
# t = tf.convert_to_tensor(5, dtype=tf.float32)
# # sys.stderr = open(log_path, mode='a', encoding='utf-8')
# # log.info(t)
# 
# tf.print(t, output_stream='file:///Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/logs/tf_test.log')



#    roi_pooling
#    将10*10切成5个2*2
# x = tf.cast(tf.random.uniform(shape=(10, 10), minval=1, maxval=10), dtype=tf.int8)
# xs = tf.split(x, [2,2,2,2,2], axis=0)           #    切成5个2*10
# res = []
# for x_ in xs:                                   #    将每个2*10切成5个2*2
#     res.append(tf.split(x_, [2,2,2,2,2], axis=1))
#     pass
# print(res)
#    将H*W roipooling 成kw*kh
# B, H, W, C = 1, 20, 20, 1
# kh, kw = 7, 7
# x = tf.cast(tf.random.uniform(shape=(B, H, W, C), minval=1, maxval=10), dtype=tf.int8)
# print(x)
# hu = round(H / kh)
# h_s = [hu for _ in range(kh)]
# h_s[-1] += H - kh * hu
# wu = round(W / kw)
# w_s = [wu for _ in range(kw)]
# w_s[-1] += W - kw * wu
# print(h_s)
# print(w_s)
# 
# #    先切H，再切W。
# x_h = tf.split(x, h_s, axis=1)
# idx_h, idx_w = 0, 0
# T = []
# for xh in x_h:
#     idx_w = 0
#     x_h = tf.split(xh, w_s, axis=2)
#     t = []
#     for xw in x_h:
#         t.append(tf.nn.max_pool2d(xw, ksize=[h_s[idx_h], w_s[idx_w]], strides=[h_s[idx_h], w_s[idx_w]], padding='VALID'))
#         idx_w += 1
#         pass
#     T.append(t)
#     idx_h += 1
#     pass
# #    拼的时候先拼W，再拼H
# res = []
# for t in T:
#     res.append(tf.concat(t, axis=2))
#     pass
# res = tf.concat(res, axis=1)
# print(res)

x = tf.random.uniform(shape=(2, 7, 7, 1))
print(x)
roi_pooling = roi_pooling.ROIPooling(name='test roi_pooling', out_size=(3, 3))
y = roi_pooling(x)
print(y)
