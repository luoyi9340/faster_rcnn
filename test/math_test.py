# -*- coding: utf-8 -*-  
'''
验证各种math函数实际用处

@author: luoyi
Created on 2020年12月31日
'''
import math
import numpy as np
import tensorflow as tf
import utils.math_expand as me


print(math.sqrt(7000))
print(math.log(133.))
print(tf.math.log(133.))
print(np.math.log(133.))
