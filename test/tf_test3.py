# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import numpy as np
import math

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


tag = 10
num = 4
a = [math.floor(tag / num) for _ in range(num)]
d = tag % num
a = [a[i] + 1 if i < d else a[i] for i in range(len(a))]
print(a)