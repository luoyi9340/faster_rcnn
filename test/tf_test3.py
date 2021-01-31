# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf
import collections as collections


a = tf.random.uniform(shape=(1, 3), minval=1, maxval=10)
a = tf.convert_to_tensor([-10,0,10], tf.float32)
a = tf.random.normal(shape=(1))
print(a)
mean, variance = tf.nn.moments(a, axes=-1) 
print(mean, variance)
a = tf.nn.batch_normalization(a,mean=mean,
                                variance=variance,
                                offset=0,
                                scale=1,
                                variance_epsilon=0)
print(a)

