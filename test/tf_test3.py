# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf
import collections as collections


y_true = tf.ones(shape=(2, 32, 9))
B = tf.math.count_nonzero(y_true[:,0,0] + 1)
print(B)