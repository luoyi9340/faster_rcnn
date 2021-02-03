# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import numpy as np

from models.layers.pooling.preprocess import roi_pooling

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


#    roi pooling
B, H, W, C = 1, 10, 10, 1
fmaps = tf.reshape(tf.range(B * H * W * C), shape=(B, H, W, C))
y_true = tf.convert_to_tensor([
                                [
                                    [0, 0,0,4,4, 1,1,1,1],
                                    [0, 5,5,9,9, 1,1,1,1],
                                ]
                              ], dtype=tf.float32)
num = y_true.shape[1]
print(tf.squeeze(fmaps))


pf = roi_pooling(fmaps, y_true, roipooling_ksize=[2,2])
print(tf.squeeze(pf))