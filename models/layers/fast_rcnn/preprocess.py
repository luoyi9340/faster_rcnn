# -*- coding: utf-8 -*-  
'''
fast_rcnn相关预处理

@author: luoyi
Created on 2021年1月27日
'''
import numpy as np
import tensorflow as tf


#    y值预处理
def preprocess_like_array(y_true, feature_map_scaling=8):
    '''y值预处理
        @param y_true: list
                        [
                            [IoU得分，proposal左上/右下点坐标(相对原图)， 所属分类索引， 标签左上点坐标/长宽],
                            [iou, xl,yl,xr,yr, vidx, x,y,w,h]
                        ]
        @param feature_map_scaling: 特征图相对原图缩放比例
        @return numpy(num, 9)
                    [
                        [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                        [vidx, xl,yl,xr,yr, tx,th,tw,th]
                    ]
    '''
    y_true = np.array(y_true)
    #    分类索引
    vidx = y_true[:,5]
    #    原图坐标换算成特征图坐标（用的时候左上下取整，右下上取整）
    xl_o = y_true[:,1]
    xl_f = xl_o / feature_map_scaling
    yl_o = y_true[:,2]
    yl_f = yl_o / feature_map_scaling
    xr_o = y_true[:,3]
    xr_f = xr_o / feature_map_scaling
    yr_o = y_true[:,4]
    yr_f = yr_o / feature_map_scaling
    #    计算t[*]
    Px = (xl_o + xr_o) / 2
    Py = (yl_o + yr_o) / 2
    Pw = np.abs(xr_o - xl_o)
    Ph = np.abs(yr_o - yl_o)
    Gx = y_true[:,6] + y_true[:,8] / 2
    Gy = y_true[:,7] + y_true[:,9] / 2
    Gw = y_true[:,8]
    Gh = y_true[:,9]
    #    计算t[x] = (G[x] - P[x]) * P[w]
    #    计算t[y] = (G[y] - P[y]) * P[h]
    #    计算t[w] = log(G[w] / P[w])
    #    计算t[h] = log(G[h] / P[h])
    tx = (Gx - Px) * Pw
    ty = (Gy - Py) * Ph
    tw = np.log(Gw / Pw)
    th = np.log(Gh / Ph)
    return np.stack([vidx, xl_f,yl_f,xr_f,yr_f, tx,ty,tw,th], axis=1)



#    拿到y_true中vidx索引对应y_pred中的预测值
def takeout_sample_array(y_true, y_pred):
    ''' 拿到y_true中vidx索引对应y_pred中的预测值
        @param y_true: tensor(batch_size, num, 9)
                                1个批次代表1张图，1个num代表一张图的1个proposal
                                [
                                    [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                                    [vidx, xl,yl,xr,yr, tx,th,tw,th]
                                    ...
                                ]
        @param y_pred: tensor(batch_size*num, 5, 42)
                                (batch_size*num, 0, 42)：每个proposal预测属于每个分类的概率
                                (batch_size*num, 1, 42)：每个proposal的d[x]
                                (batch_size*num, 2, 42)：每个proposal的d[y]
                                (batch_size*num, 3, 42)：每个proposal的d[w]
                                (batch_size*num, 4, 42)：每个proposal的d[h]
        @return: (arrs, total, B, num) 
                        arrs: tensor(batch_size*num, 5)
                                [分类概率，dx, dy, dw, dh]
                        total: 总proposal数
                        B: batch_size数量    
                        num: 每个batch_size中包含多少记录
    '''
    y_pred = tf.transpose(y_pred, perm=[0,2,1])                 #    1个42*5代表y_true1条记录
    B = tf.math.count_nonzero(y_true[:,0,0] + 1)                #    batch_size数
    num = y_true.shape[1]                                       #    每个batch_size中的proposal数
    total = B * num                                             #    总proposal数
    
    vidx = y_true[:,:,0]
    vidx = tf.reshape(vidx, shape=(total,))
    vidx = tf.cast(vidx, dtype=tf.int32)
    idx_ = tf.range(total, dtype=tf.int32)
    idxes = tf.stack([idx_, vidx], axis=1)
    arrs = tf.gather_nd(y_pred, indices=idxes)  
    
    return (arrs, total, B, num)


