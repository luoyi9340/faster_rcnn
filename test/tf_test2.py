# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月27日
'''
import tensorflow as tf

import utils.math_expand as me
from models.layers.fast_rcnn.preprocess import takeout_sample_array
from models.layers.fast_rcnn.metrics import FastRcnnMetricCls, FastRcnnMetricReg
from models.layers.fast_rcnn.losses import FastRcnnLoss


# [vidx, xl,yl,xr,yr, tx,th,tw,th]
y_true = tf.convert_to_tensor([
                                [
                                    [0, 0,0,10,10, 1,1,1,1],
                                    [1, 1,1,11,11, 1,1,1,1],
                                    [2, 1,1,11,11, 1,1,1,1],
                                ],
                                [
                                    [1, 1,1,11,11, 1,1,1,1],
                                    [2, 0,0,10,10, 1,1,1,1],
                                    [3, 1,1,11,11, 1,1,1,1],
                                ]
                              ], dtype=tf.float32)
B, num, C = y_true.shape[0], y_true.shape[1], 4
total = B * num
y_pred_cls = tf.random.uniform(shape=(B*num, 1, C))
y_pred_cls = tf.nn.softmax(y_pred_cls)
y_pred_reg = tf.random.uniform(shape=(B*num, 4, C))
y_pred = tf.concat([y_pred_cls, y_pred_reg], axis=1)

(arrs, total, B, num) = takeout_sample_array(y_true, y_pred)
print(arrs)

fast_rcnn_loss = FastRcnnLoss()
loss = fast_rcnn_loss(y_true, y_pred)
print(loss)

