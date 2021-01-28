# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np

from models.layers.fast_rcnn.preprocess import preprocess_like_array


#    print时不用科学计数法表示
np.set_printoptions(suppress=True)     


y_true = [
                [0.9, 5,5,15,15, 0, 5,5,10,10],
                [0.9, 10,10,20,20, 0, 10,10,10,10],
                [0.9, 15,15,25,25, 0, 15,15,10,10],
                [0.9, 20,20,30,30, 0, 20,20,10,10]
            ]
p = preprocess_like_array(y_true)
print(p)