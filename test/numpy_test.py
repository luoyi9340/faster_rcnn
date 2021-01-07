# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np


#    [IoU, [x(中心点), y(中心点), w, h, idx_w, idx_h, idx_area, idx_scales], ["验证码值", x(左上点), y(左上点), w, h]]
# a = [
#         [-1, [0,0,0,0,0,0,0,0]],
#         [-1, [0,0,0,0,0,0,0,0]]
#     ]
# b = [
#         ['', 0,0,0,0],
#         ['', 0,0,0,0]
#     ]
# a = np.c_[a, b]
# print(a)
# if (len(a) < 10):
#     a = a + [[-1, [0,0,0,0,0,0,0,0], ['', 0,0,0,0]] for i in range(10 - len(a))]
# print(len(a))


# a = [[1,1,1]]
# b = [[2,2,2]]
# print(a + b)


# a = np.arange(start=0.1, stop=1, step=0.1)
# b = np.arange(start=0.1, stop=0.5, step=0.1)
# a = np.hstack([a, b])
# print(a)
# print(np.count_nonzero((1-a) > 0.7))


#    自定义的loss里没法用numpy，只能fit之前根据y_true整一个跟feature_maps尺寸一样的张量，loss里直接跟y_pred运算（类似one_hot）
# zt = np.zeros(shape=(3,3,3))
# #    (0,0,0) = 1, (1,1,1)=2, (2,2,2)=3
# # zt[0,0,0] = 1
# # zt[1,1,1] = 2
# # zt[2,2,2] = 3
# idx = [[0,0,0], [1,1,1], [2,2,2]]
# a1, a2, a3 = zip(*idx)
# print(a1)
# print(a2)
# print(a3)
# a1 = tuple([0,1,2])
# a2 = tuple([0,1,2])
# a3 = tuple([0,1,2])
# zt[a1, a2, a3] = [1,2,3]
# print(zt)
feature_maps =np.random.uniform(size=(4,4,6,4), low=0, high=100).astype(np.int8)
y_true = [
            [0.75, 1, 2, 10, 10, 0, 0, 0, 0, 15, 0, 0, 10, 10],            #    (x,y)=(0,0) c=0*2+0=0
            [0.75, 1, 2, 10, 10, 1, 1, 0, 1, 15, 10, 10, 10, 10],          #    (x,y)=(1,1) c=0*2+1=1
            [0.25, 1, 2, 10, 10, 2, 2, 1, 0, 15, 20, 20, 10, 10],          #    (x,y)=(2,2) c=1*2+0=2
            [0.25, 1, 2, 10, 10, 3, 3, 1, 1, 15, 30, 30, 10, 10]           #    (x,y)=(2,3) c=1*2+1=3
        ]
y_true = np.array(y_true)
y_true_positives = y_true[(y_true[:,0] > 0.7)]
y_true_negative = y_true[(y_true[:,0] < 0.3)]
# idx_fmap = y_true[:, [5,6]]
# idx_anchor = y_true[:, [7,8]]
zero_template = np.zeros(shape=(4, 4, 6, 4))
#    正样本的前景概率置为1
idx_fmap = y_true_positives[:, [5,6]].astype(np.int8)
idx_anchor = y_true_positives[:, [7,8]].astype(np.int8)
idx_anchor = idx_anchor[:,0] * 2 + idx_anchor[:,1]
idx_anchor = tuple(idx_anchor)
fmap_w = tuple(idx_fmap[:,0])
fmap_h = tuple(idx_fmap[:,1])
zero_template[fmap_w, fmap_h, 0, idx_anchor] = 1
#    负样本的背景规律置为1
idx_fmap = y_true_negative[:, [5,6]].astype(np.int8)
idx_anchor = y_true_negative[:, [7,8]].astype(np.int8)
idx_anchor = idx_anchor[:,0] * 2 + idx_anchor[:,1]
idx_anchor = tuple(idx_anchor)
fmap_w = tuple(idx_fmap[:,0])
fmap_h = tuple(idx_fmap[:,1])
zero_template[fmap_w, fmap_h, 1, idx_anchor] = 1
#    计算所有正样本的t[*]
Gx = y_true_positives[:,10] + y_true_positives[:,12]/2
Gy = y_true_positives[:,11] + y_true_positives[:,13]/2
Gw = y_true_positives[:,12]
Gh = y_true_positives[:,13]
Px = y_true_positives[:,1]
Py = y_true_positives[:,2]
Pw = y_true_positives[:,3]
Ph = y_true_positives[:,4]
Tx = (Gx - Px) * Pw         #    计算t[x] = (G[x] - P[x]) * P[w]
Ty = (Gy - Py) * Ph         #    计算t[y] = (G[y] - P[y]) * P[h]
Tw = np.log(Gw / Pw)        #    计算t[w] = log(G[w] / P[w])
Th = np.log(Gh / Ph)        #    计算t[h] = log(G[h] / P[h])
#    根据fmap索引和anchor索引写入模板
idx_fmap = y_true_positives[:, [5,6]].astype(np.int8)
idx_anchor = y_true_positives[:, [7,8]].astype(np.int8)
idx_anchor = idx_anchor[:,0] * 2 + idx_anchor[:,1]
idx_anchor = tuple(idx_anchor)
fmap_w = tuple(idx_fmap[:,0])
fmap_h = tuple(idx_fmap[:,1])
print(fmap_w, fmap_h, idx_anchor)
print(Tx, Ty, Tw, Th)
zero_template[fmap_w, fmap_h, 2, idx_anchor] = Tx
zero_template[fmap_w, fmap_h, 3, idx_anchor] = Ty
zero_template[fmap_w, fmap_h, 4, idx_anchor] = Tw
zero_template[fmap_w, fmap_h, 5, idx_anchor] = Th
print(zero_template)
