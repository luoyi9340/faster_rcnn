# -*- coding: utf-8 -*-  
'''
函数补充

@author: luoyi
Created on 2021年1月4日
'''
import math
import numpy as np
import tensorflow as tf


#    smootL1函数
def smootL1_np(x):
    '''smootL1函数, numpy版本
        smootL1(x) =  x² * 0.5    x∈(0,1)
                     |x| - 0.5    x∈其他
    '''
    x_abs = np.abs(x)
    
    x_0 = np.zeros_like(x)
    x_0[x_abs < 1] = x[x_abs < 1]
    x_0 = np.power(x_0, 2) * 0.5

    x_1 = np.zeros_like(x)
    x_1[x_abs >= 1] = x[x_abs >= 1]
    x_1 = np.fabs(x_1) - 0.5
    x_1[x_abs < 1] = 0
    
    return x_0 + x_1


#    smootL1函数
def smootL1_tf(x):
    '''smootL1函数, numpy版本
        smootL1(x) =  x² * 0.5    x∈(0,1)
                     |x| - 0.5    x∈其他
    '''
    x_abs = tf.abs(x)
    x_pow2 = tf.pow(x, 2)
    return tf.where(x_abs < 1, x_pow2 * 0.5, x_abs - 0.5)


#    smootL1函数
def smootL1(x):
    return math.pow(x, 2) * 0.5 if math.fabs(x) < 1 else math.fabs(x) - 0.5


#    相对均匀分配数值给列表
def fairly_equalize(tag, num):
    '''要把tag均匀分成num份，每份该是多少（只能是整数）
    '''
    a = [math.floor(tag / num) for _ in range(num)]
    d = tag % num
    a = [a[i] + 1 if i < d else a[i] for i in range(len(a))]
    return a

