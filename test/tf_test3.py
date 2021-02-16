# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf
import numpy as np
import math

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


a = tf.reshape(tf.range(27), shape=(3,3,3))
print(a)
a = tf.math.reduce_mean(a, axis=(1,2))
print(a)


