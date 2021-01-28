# -*- coding: utf-8 -*-  
'''
数据源处理相关零件

@author: luoyi
Created on 2021年1月24日
'''
import PIL
import numpy as np
import math


#    读取图片矩阵
def read_image(image_path='', resize_weight=None, resize_height=None, preprocess=None):
    '''读取图片矩阵
        @param image_path: 图片路径
        @param resize_weight: 统一大小宽度
        @param resize_height: 统一大小高度
        @param preprocess: 后置处理。参数：(3, h, w)
    '''
    image = PIL.Image.open(image_path, mode='r')
    image = image.resize((resize_weight, resize_height),PIL.Image.ANTIALIAS)
    #        重置大小
    if ((resize_weight is not None) and (resize_height is not None)):
        x = np.asarray(image, dtype=np.float32)
        pass
    #    x数据做前置处理（归一化在这里做）
    if (preprocess is not None):x = preprocess(x)
    return x


#    IoU计算
def iou_center_wh(rect1=None, rect2=None):
    '''IoU计算
        @param rect1: (x中心点坐标, y中心点坐标, w宽度, h高度)
        @param rect2: (x中心点坐标, y中心点坐标, w宽度, h高度)
    '''
    r1x, r1y, r1w, r1h = rect1[0], rect1[1], rect1[2], rect1[3]
    r2x, r2y, r2w, r2h = rect2[0], rect2[1], rect2[2], rect2[3]
        
    #    根据label和anchor中心点坐标判断位置关系：相离 / 相交
    #    如果两个中心坐标距离超过两个半径之和，则判定为相离，直接返回0
    if (math.fabs(r1x - r2x)  > (r1w + r2w) / 2.) \
        or (math.fabs(r1y - r2y)  > (r1h + r2h) / 2.):
        return 0.
    #    计算交集面积(4个顶点x中位于中间的两个 和 4个顶点y中位于中间的两个就是交集坐标)
    arr_x = [r1x + r1w/2, r1x - r1w/2, r2x + r2w/2, r2x - r2w/2]
    arr_x.sort()
    arr_y = [r1y + r1h/2, r1y - r1h/2, r2y + r2h/2, r2y - r2h/2]
    arr_y.sort()
    area_intersection = math.fabs(arr_x[1] - arr_x[2]) * math.fabs(arr_y[1] - arr_y[2])
    #    计算并集面积
    area_union = r1w * r1h + r2w * r2h - area_intersection
    return area_intersection / area_union

#    IoU计算
def iou_xlyl_xryr_np(rect_srcs=None, rect_tag=None):
    '''IoU计算
        @param rect_srcs: numpy(num, 4)
                            [
                                [xl, yl, xr, yr]
                                ...
                            ]
        @param rect_tag: tuple(4) 
                            (xl, yl, xr, yr)
        @return: IoU numpy(num, 1)
    '''
    #    取交点处坐标
    xl = np.maximum(rect_tag[0], rect_srcs[:,0], dtype=np.float32)
    yl = np.maximum(rect_tag[1], rect_srcs[:,1], dtype=np.float32)
    xr = np.minimum(rect_tag[2], rect_srcs[:,2], dtype=np.float32)
    yr = np.minimum(rect_tag[3], rect_srcs[:,3], dtype=np.float32)
    #    取交点区域长宽
    w = np.maximum(0., xr - xl)
    h = np.maximum(0., yr - yl)
    #    计算tag, srcs面积
    area_tag = np.abs(rect_tag[0] - rect_tag[2]) * np.abs(rect_tag[1] - rect_tag[3])
    area_srcs = np.abs(rect_srcs[:, 0] - rect_srcs[:, 2]) * np.abs(rect_srcs[:, 1] - rect_srcs[:, 3])
    areas_intersection = w * h
    areas_union = (np.add(area_tag, area_srcs) - areas_intersection).astype(np.float32)
    IoU = (areas_intersection / areas_union).astype(np.float32)
    return IoU


