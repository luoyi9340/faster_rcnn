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


#    验算：preprocess_like_fmaps中t[*]的反向计算应该能还原label的x,y,w,h
# IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x, y, w, h
Y = [
        [0.9, 10,10,20,20, 0,0,0,0, 1, 11,11,22,22],
#         [0.9, 20,20,40,40, 0,0,0,0, 1, 21,21,42,42]
    ]
Y = np.array(Y)
Y_maps = preprocess_like_fmaps(Y, shape=(23, 60, 6, 15))
#    还原label的x,y,w,h
Tx = Y_maps[:,:, 2, :]
Tx = Tx[Tx > 0]
Ty = Y_maps[:,:, 3, :]
Ty = Ty[Ty > 0]
Tw = Y_maps[:,:, 4, :]
Tw = Tw[Tw > 0]
Th = Y_maps[:,:, 5, :]
Th = Th[Th > 0]
Px = Y[:, 1]
Py = Y[:, 2]
Pw = Y[:, 3]
Ph = Y[:, 4]
#    G[x] = t[x]/P[w] + P[x]
#    G[y] = t[y]/P[h] + P[y]
#    G[w] = exp(t[w]) * P[w]
#    G[h] = exp(t[h]) * P[h]
Gx = Tx / Pw + Px
Gy = Ty / Ph + Py
Gw = np.exp(Tw) * Pw
Gh = np.exp(Th) * Ph
print(Gx.shape, Gx - Gw/2)
print(Gy.shape, Gy - Gh/2)
print(Gw.shape, Gw)
print(Gh.shape, Gh)


