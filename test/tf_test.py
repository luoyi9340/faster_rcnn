# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import numpy as np


#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


class TestLayer(tf.keras.layers.Layer):
    def call(self, x, y, **kwargs):
        print(x)
        print(y)
        return x
    pass


x = tf.ones(shape=[1,1])
y = tf.zeros(shape=[1,1])

tl = TestLayer()
tl(x=x, y=y)


