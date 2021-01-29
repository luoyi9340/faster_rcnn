# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf


class TL(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TL, self).__init__(**kwargs)
        pass
    def call(self, x, y, **kwargs):
        return x + y
    pass

tl = TL()
y = tl(x=1, y=2)
print(y)