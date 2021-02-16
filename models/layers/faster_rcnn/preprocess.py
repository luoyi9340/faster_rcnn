# -*- coding: utf-8 -*-  
'''
相关函数

@author: luoyi
Created on 2021年2月15日
'''
import tensorflow as tf

import utils.conf as conf


#    通过原始labels生成rois
def create_rois_from_labels(labels=None,
                            positives_iou=conf.ROIS.get_positives_iou(),
                            negative_iou=conf.ROIS.get_negative_iou()):
    '''
        @param labels: 原始标签 tensor(6, 5)
                        [vidx, x,y, w,h] 相对原图
        @param positives_iou: 正样本IoU阈值
        @param negative_iou: 负样本IoU阈值
    '''
    
    
    pass
