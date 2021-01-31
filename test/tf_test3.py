# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf
import collections as collections
import numpy as np

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


a = tf.random.normal(shape=(1, 100000), mean=0, stddev=10)
# print(a)
mean, variance = tf.nn.moments(a, axes=-1) 
print(mean, variance)
a = tf.nn.batch_normalization(a,mean=mean,
                                variance=variance,
                                offset=0,
                                scale=1,
                                variance_epsilon=0)
mean, variance = tf.nn.moments(a, axes=-1) 
print(mean, variance)
print(a[a > 1])

