# -*- coding: utf-8 -*-  
'''
loss函数测试

@author: luoyi
Created on 2021年1月23日
'''
import numpy as np
import tensorflow as tf

from models.layers.rpn.preprocess import preprocess_like_array, takeout_sample_array
from models.layers.rpn.losses import RPNLoss


B, H, W, K = 2, 4, 4, 4

#    假设40*40的原图，缩放比例10，特征图4*4
roi_areas = np.array([10, 20])
roi_scales = np.array([1, 2])
#    [IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x(label左上点), y(label左上点), w, h]
y_true = [
            [0.9, 5,5,10,10, 0,0,0,0, 1, 0,10,10,15],
            [0.9, 15,15,40,10, 1,1,1,1, -1, 10,35,40,10],
            [0.9, 25,35,20,20, 2,2,1,0, 1, 15,15,20,20],
            [0.9, 35,35,20,5, 3,3,0,1, -1, 37.5,45,20,5]
        ]
y_true = np.array(y_true)
y = preprocess_like_array(y_true)
y = np.expand_dims(y, axis=0)
y = np.repeat(y, 2, axis=0)
#    模拟标签数据[batch_size=4, num=4, 10]
y_true = tf.convert_to_tensor(y, dtype=tf.float32)

fmaps_cls = tf.random.uniform(shape=(B, H, W, 2, K))
fmaps_cls = tf.nn.softmax(fmaps_cls, axis=3)
fmaps_cls = tf.transpose(fmaps_cls, perm=[0,1,2,4,3])
fmaps_reg = tf.range(B*H*W*K*4, dtype=tf.float32)
fmaps_reg = tf.reshape(fmaps_reg, shape=(B, H, W, K, 4))
fmaps = tf.concat([fmaps_cls, fmaps_reg], axis=-1)
fmaps = tf.transpose(fmaps, perm=[0,1,2,4,3])

print(y_true)
anchors = takeout_sample_array(y_true, fmaps)
fmaps = tf.transpose(fmaps, perm=[0,1,2,4,3])
# print(fmaps)
print(anchors)

rpn_loss = RPNLoss()
loss_reg = rpn_loss.loss_reg(anchors, y_true)
print(loss_reg)

