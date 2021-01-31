# -*- coding: utf-8 -*-  
'''
roi_pooling 测试

@author: luoyi
Created on 2021年1月31日
'''
import tensorflow as tf

from models.layers.pooling.preprocess import roi_align


#    模拟10*10特征图
H, W = 10, 10
# fmaps = tf.range(H * W)
# fmaps = tf.reshape(fmaps, shape=(1, H, W, 1))
fmaps1 = tf.convert_to_tensor([
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [2, 2, 2, 3, 4, 5, 6, 7, 8, 9],
                            [3, 3, 3, 3, 4, 5, 6, 7, 8, 9],
                            [4, 4, 4, 4, 4, 5, 6, 7, 8, 9],
                            [5, 5, 5, 5, 5, 5, 6, 7, 8, 9],
                            [6, 6, 6, 6, 6, 6, 6, 7, 8, 9],
                            [7, 7, 7, 7, 7, 7, 7, 7, 8, 9],
                            [8, 8, 8, 8, 8, 8, 8, 8, 8, 9],
                            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                        ])
fmaps1 = tf.expand_dims(fmaps1, axis=0)
fmaps1 = tf.expand_dims(fmaps1, axis=-1)
fmaps2 = fmaps1 * 2
fmaps3 = fmaps1 * 3
fmaps = tf.concat([fmaps1, fmaps2, fmaps3], axis=0)

y_true = tf.convert_to_tensor([
                            [[0, 0,0,4,4, 1,1,1,1],
                             [0, 5,0,9,4, 1,1,1,1],
                             [0, 0,5,4,9, 1,1,1,1],
                             [0, 5,5,9,9, 1,1,1,1]],
                            [[0, 0,0,4,4, 1,1,1,1],
                             [0, 5,0,9,4, 1,1,1,1],
                             [0, 0,5,4,9, 1,1,1,1],
                             [0, 5,5,9,9, 1,1,1,1]],
                            [[0, 0,0,4,4, 1,1,1,1],
                             [0, 5,0,9,4, 1,1,1,1],
                             [0, 0,5,4,9, 1,1,1,1],
                             [0, 5,5,9,9, 1,1,1,1]]
                        ], dtype=tf.float32)
crops = roi_align(fmaps, y_true, roipooling_ksize=[2,2])

fmaps = tf.squeeze(fmaps)
print(fmaps)

crops = tf.squeeze(crops)
print(crops)

