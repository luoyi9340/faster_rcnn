# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np
import math

feature_map_scaling = 8
IMAGE_HEIGHT = 180
IMAGE_WEIGHR = 480
roi_areas = np.array([64, 68])
roi_scales = np.array([0.75, 1])
B, H, W, K = 2, 2, 2, 4


a = np.random.uniform(size=[5, 5], low=-1, high=1)
print(a)
a[(a[:,1] < -0.1) + (a[:,1] > 0.1)] = [-1, 0,0,0,0]      
print(a)               
                     
                     
                     
                     
