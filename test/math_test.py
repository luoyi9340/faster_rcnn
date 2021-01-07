# -*- coding: utf-8 -*-  
'''
验证各种math函数实际用处

@author: luoyi
Created on 2020年12月31日
'''
import math
import numpy as np
import utils.math_expand as me


#    检测loss_cls算的对不对
#    分类loss
# l1 = math.log(0.40134785)
# l2 = math.log(0.71501285)
# l3 = math.log(0.7312956)
# l4 = math.log(0.2668086)
# mean = (l1 + l2 + l3 + l4) / 4.
# print(l1, l2, l3, l4)
# print(mean)

#    回归loss
# y = 17.07648
# f = 25.405191
# print(y - f)
# loss = me.smootL1(y - f)
# print(loss)
v = [15.892754, 22.423492, 8.592064, 2.077179, 3.9427366, 9.163124, 11.612677, 8.563061, 32.494835, 27.48685, 11.758651, 34.368156, 14.096106, 4.6647234, 17.268024, 44.83168]
v = np.array(v)
print(np.mean(v))

