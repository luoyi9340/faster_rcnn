# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf


t = tf.ones(shape=[2, 2,2, 3])
print(t)
t = tf.pad(t, [[0,0], [1,1],[1,1], [0,0]])
print(t)

