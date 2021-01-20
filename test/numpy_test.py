# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np
import math

import data.dataset_rois as rois
from models.layers.rpn.preprocess import all_positives_from_fmaps, preprocess_like_fmaps       

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)     


#    融合两个数组
a = np.array([[1,2,3],
              [1,2,3],
              [1,2,3]])
b = np.array([[4,5,6],
              [4,5,6],
              [4,5,6]])
c = [np.concatenate([np.expand_dims(a_, axis=0), np.expand_dims(b_, axis=0)]) for a_, b_ in zip(a, b)]
c = np.concatenate(c, axis=0)
c[:,0] = -c[:,0]
print(c)