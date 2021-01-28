# -*- coding: utf-8 -*-  
'''
非极大值抑制
    过滤掉fmaps中重叠的anchor

@author: luoyi
Created on 2021年1月9日
'''
import numpy as np

import utils.conf as conf
import data.part as part


#    非极大值抑制
def nms(anchors, threshold=conf.RPN.get_nms_threshold_iou()):
    '''非极大值抑制
        step1: anchors按正样本概率倒序
        step2: 取anchors[0]正样本概率最大的anchor，并从anchors中剔除，计算anchors中其他anchor与其的IoU
        step3: IoU超过threshold的直接从anchors中剔除
        step4: 重复step2直到anchors为空。每次取得的anchors[0]就是非极大值抑制的结果
        
        @param anchors: [narray(num, 5)...] 全部判定为正样本的anchors
                            最后8位含义：正样本概率, xl, yl, xr, yr)
        @param threshold: 阈值
        @return: anchors [narray(num, 6)...]
                            [正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
    '''
    res = []
    for a in anchors:
        t = _nms(a, threshold=threshold)
        res.append(t)
        pass
    return res



#    单图片的候选框NMS
def _nms(anchors, 
         threshold=conf.RPN.get_nms_threshold_iou()):
    '''单图片的候选框非极大值抑制
        @param anchor: 单图片生成的候选框narray(num, 5)
                        最后8位含义：正样本概率, xl,yl, xr,yr
        @param feature_map_scaling: 阈值（超过此阈值的IoU会被判定为重叠而过滤掉）
        @return narray(num, 6)
                    [正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
    '''
    #    anchors按anchors[:,0]降序
    anchors = anchors[np.argsort(anchors[:,0])[::-1]]
    #    收集计算需要用到的数据[正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
    anchors_tmp = np.zeros(shape=(anchors.shape[0], 6))
    #    正样本概率
    anchors_tmp[:,0] = anchors[:,0]
    anchors_tmp[:,1] = anchors[:,1]
    anchors_tmp[:,2] = anchors[:,2]
    anchors_tmp[:,3] = anchors[:,3]
    anchors_tmp[:,4] = anchors[:,4]
    #    计算每个anchor自身的面积
    w = np.abs(anchors[:,3] - anchors[:,1])
    h = np.abs(anchors[:,4] - anchors[:,2])
    anchors_tmp[:,5] = w * h
    anchors_tmp.astype(np.float32)

    #    非极大值抑制
    nms = [] 
    while (len(anchors_tmp) > 0):
        max_prob_anchor = anchors_tmp[0]
        nms.append(np.expand_dims(max_prob_anchor, axis=0))
        anchors_tmp = np.delete(anchors_tmp, 0, axis=0)
        
        if (len(anchors_tmp) == 0): break
        #    计算IoU
        rect_tag = (max_prob_anchor[1], max_prob_anchor[2], max_prob_anchor[3], max_prob_anchor[4])
        rect_srcs = np.zeros(shape=(anchors_tmp.shape[0], 4))
        rect_srcs[:,0] = anchors_tmp[:,1]
        rect_srcs[:,1] = anchors_tmp[:,2]
        rect_srcs[:,2] = anchors_tmp[:,3]
        rect_srcs[:,3] = anchors_tmp[:,4]
        IoU = part.iou_xlyl_xryr_np(rect_srcs, rect_tag)

        idx = np.where(IoU <= threshold)
        #    剔除掉IoU>阈值的部分
        anchors_tmp = anchors_tmp[idx]
        pass
    return np.concatenate(nms, axis=0)
