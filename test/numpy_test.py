# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np
import math

import data.dataset_rois as rois
from models.layers.rpn.preprocess import all_positives_from_fmaps, preprocess_like_fmaps, preprocess_like_array   

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)     



