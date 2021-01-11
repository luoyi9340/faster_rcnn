# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月1日
'''
import numpy as np
import math


IMAGE_HEIGHT = 180
IMAGE_WEIGHR = 480
B, H, W, K = 2, 2, 2, 4

a = np.random.uniform(size=(B,H,W, 6, K))
[p,n,reg] = np.split(a, [1,2], axis=3)
p[p <= 0.5] = 0

tmp = np.zeros_like(p)
tmp[p > 0.5] = 1
tmp = np.repeat(tmp, K, axis=3)
reg = tmp * reg

a = np.concatenate([p, reg], axis=3)        #    a.shape=(batch_size, h, w, 5, K)。a[b,h,w]的每列代表一个anchor，每列数据为[prob, d[x], d[y], d[w], d[h]]
a = np.transpose(a, axes=(0,1,2, 4,3))      #    a.shape=(batch_size, h, w, K, 5)。a[b,h,w]的每行代表一个anchor，每行数据为[prob, d[x], d[y], d[w], d[h]]
#    给a的每行追加h[0,H), w[0,W), anchor[0,K)索引。扩充后a.shape=(2,2,2, 4, 8)。a[b,h,w]每行数据为[prob, d[x], d[y], d[w], d[h], idx_h, idx_w, idx_anchor]
idx_anchor = np.arange(K)                   #    得到单位anchor索引shape=(K)
idx_anchor = np.concatenate([idx_anchor for _ in range(H*W)])
idx_anchor = np.reshape(idx_anchor, newshape=(len(idx_anchor), 1))
idx_w, idx_h = np.meshgrid(range(W), range(H))
idx_w = np.reshape(idx_w, newshape=H*W)
idx_h = np.reshape(idx_h, newshape=H*W)
idx = np.concatenate([np.expand_dims(idx_h, axis=0), np.expand_dims(idx_w, axis=0)], axis=0)
idx = np.transpose(idx, axes=(1, 0))        #    得到单位h,w索引 shape=(H*W, 2)
idx = np.repeat(idx, K, axis=0)             #    (H*W, 2)扩展为(H*W*K, 2)
idx = np.concatenate([idx, idx_anchor], axis=1)
idx = np.reshape(idx, newshape=(H, W, K, 3))
idx = np.repeat(np.expand_dims(idx, axis=0), B, axis=0)
a = np.concatenate([a, idx], axis=4)

#    过滤掉概率为0的
a = a[a[:,:,:,:,0] > 0]
#    按概率降序
a = a[np.argsort(a[:,0])[::-1]]
print(a)

#    NMS
#    保存anchor. [前景概率, xl,yl(左上), xr,yr(右下), 面积]
anchors = np.zeros(shape=(a.shape[0], 6))
anchors[:,0] = a[:,0]
#    按照原图[180*480], 特征图的缩放比8, roi_area=[64,68,72,76,84] roi_scales=[0.75, 1, 1.25]还原[xl,yl,xr,yr]
feature_map_scaling = 8
IMAGE_HEIGHT = 180
IMAGE_WEIGHR = 480
roi_area = np.array([64, 68])
roi_scales = np.array([0.75, 1])
#    还原左上点坐标
anchors[:,1] = a[:,5] * feature_map_scaling
anchors[:,2] = a[:,6] * feature_map_scaling
#    还原右下点坐标
# print(a[:,7].astype(np.int8))
# print(roi_area[(a[:,7] / len(roi_scales)).astype(np.int8)])
# print(roi_scales[(a[:,7] % len(roi_scales)).astype(np.int8)])
#    右下点坐标 = 左上点坐标 + 宽高
anchors[:,3] = anchors[:,1] + np.around(roi_area[(a[:,7] / len(roi_scales)).astype(np.int8)] * roi_scales[(a[:,7] % len(roi_scales)).astype(np.int8)])
anchors[:,4] = anchors[:,2] + np.around(roi_area[(a[:,7] / len(roi_scales)).astype(np.int8)] / roi_scales[(a[:,7] % len(roi_scales)).astype(np.int8)])
#    计算每个anchor自身的面积
anchors[:,5] = np.abs(anchors[:,3] - anchors[:,1]) * np.abs(anchors[:,4] - anchors[:,2])
anchors.astype(np.float32)
#    非极大值抑制
nms = [] 
threshold = 0.99
while (len(anchors) > 0):
    max_prob_anchor = anchors[0]
    nms.append(max_prob_anchor)
    anchors = np.delete(anchors, 0, axis=0)
    
    if (len(anchors) == 0): break

    #    计算max与其他anchor的IoU
    #    计算交点处的左上、右下坐标
    xl = np.maximum(max_prob_anchor[1], anchors[:,1], dtype=np.float32)
    yl = np.maximum(max_prob_anchor[2], anchors[:,2], dtype=np.float32)
    xr = np.minimum(max_prob_anchor[3], anchors[:,3], dtype=np.float32)
    yr = np.minimum(max_prob_anchor[4], anchors[:,4], dtype=np.float32)
    #    计算交集、并集面积
    w = np.maximum(0., np.abs(xr - xl))
    h = np.maximum(0., np.abs(yr - yl))
    areas_intersection = w * h
    areas_union = (np.add(max_prob_anchor[5], anchors[:,5]) - areas_intersection).astype(np.float32)
    IoU = (areas_intersection / (areas_union + areas_intersection)).astype(np.float32)
    print(areas_intersection)
    print(areas_union)
    print(IoU)
    print()
    #    剔除掉IoU>阈值的部分
    anchors = anchors[np.where(IoU <= threshold)]
    pass
nms = np.array(nms)
print(nms[:,0], nms[:,1:])



