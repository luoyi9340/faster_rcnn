# -*- coding: utf-8 -*-  
'''
非极大值抑制
    过滤掉fmaps中重叠的anchor

@author: luoyi
Created on 2021年1月9日
'''
import numpy as np

import utils.conf as conf


#    非极大值抑制
def nms(anchors, 
                threshold=conf.RPN.get_nms_threshold_iou(), 
                feature_map_scaling=conf.CNNS.get_feature_map_scaling(),
                roi_areas=conf.RPN.get_roi_areas(),
                roi_scales=conf.RPN.get_roi_scales()
                ):
    '''非极大值抑制
        step1: anchors按正样本概率倒序
        step2: 取anchors[0]正样本概率最大的anchor，并从anchors中剔除，计算anchors中其他anchor与其的IoU
        step3: IoU超过threshold的直接从anchors中剔除
        step4: 重复step2直到anchors为空。每次取得的anchors[0]就是非极大值抑制的结果
        
        @param anchors: numpy(num, 8) 全部判定为正样本的anchors
                            最后8位含义：正样本概率, d[x], d[y], d[w], d[h], idx_h(相对于特征图) idx_w(相对于特征图), idx_andhor(area * scales索引)
        @param threshold: 阈值
        @return: anchors(numpy)
                    [正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
    '''
    #    anchors按anchors[:,0]降序
    anchors = anchors[np.argsort(anchors[:,0])[::-1]]
    #    收集计算需要用到的数据[正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
    anchors_tmp = np.zeros(shape=(anchors.shape[0], 6))
    #    正样本概率
    anchors_tmp[:,0] = anchors[:,0]
    #    还原左上点坐标
    anchors_tmp[:,1] = anchors[:,5] * feature_map_scaling
    anchors_tmp[:,2] = anchors[:,6] * feature_map_scaling
    #    右下点坐标 = 左上点坐标 + 宽高
    anchors_tmp[:,3] = anchors[:,1] + np.around(roi_areas[(anchors[:,7] / len(roi_scales)).astype(np.int8)] * roi_scales[(anchors[:,7] % len(roi_scales)).astype(np.int8)])
    anchors_tmp[:,4] = anchors[:,2] + np.around(roi_areas[(anchors[:,7] / len(roi_scales)).astype(np.int8)] / roi_scales[(anchors[:,7] % len(roi_scales)).astype(np.int8)])
    #    计算每个anchor自身的面积
    anchors_tmp[:,5] = np.abs(anchors_tmp[:,3] - anchors_tmp[:,1]) * np.abs(anchors_tmp[:,4] - anchors_tmp[:,2])
    anchors_tmp.astype(np.float32)

    #    非极大值抑制
    nms = [] 
    while (len(anchors_tmp) > 0):
        max_prob_anchor = anchors_tmp[0]
        nms.append(max_prob_anchor)
        anchors_tmp = np.delete(anchors_tmp, 0, axis=0)
        
        if (len(anchors_tmp) == 0): break
    
        #    计算max与其他anchor的IoU
        #    计算交点处的左上、右下坐标
        xl = np.maximum(max_prob_anchor[1], anchors_tmp[:,1], dtype=np.float32)
        yl = np.maximum(max_prob_anchor[2], anchors_tmp[:,2], dtype=np.float32)
        xr = np.minimum(max_prob_anchor[3], anchors_tmp[:,3], dtype=np.float32)
        yr = np.minimum(max_prob_anchor[4], anchors_tmp[:,4], dtype=np.float32)
        #    计算交集、并集面积
        w = np.maximum(0., np.abs(xr - xl))
        h = np.maximum(0., np.abs(yr - yl))
        areas_intersection = w * h
        areas_union = (np.add(max_prob_anchor[5], anchors_tmp[:,5]) - areas_intersection).astype(np.float32)
        IoU = (areas_intersection / (areas_union + areas_intersection)).astype(np.float32)
        #    剔除掉IoU>阈值的部分
        anchors = anchors_tmp[np.where(IoU <= threshold)]
        pass
    nms = np.array(nms)
    return np.array(nms)

