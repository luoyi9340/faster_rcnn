# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf


#    roipooling
def roi_pooling(fmaps, y_true, roipooling_ksize=[7, 7]):
    '''roi pooling
        @param fmaps: tensor(batch_size, H, W, C)
        @param y_true: tensor(batch_size, num, 9)
                        [
                            [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                            [vidx, xl,yl,xr,yr, tx,th,tw,th]
                        ]
        @return: tensor(batch_size, num, roipooling_ksize[0], roipooling_ksize[1], C)
    '''
    #    取必要的参数
    H, W = fmaps.shape[1], fmaps.shape[2]                                   #    特征图宽高
    B, num = y_true.shape[0], y_true.shape[1]                               #    batch_size, 每个batch_size中的proposal数
    crop_size = [roipooling_ksize[0] * 2, roipooling_ksize[1] * 2]          #    crop_and_resize的ksize
    
    #    将特征图统一切成roipooling_ksize * 2的大小
    xl = y_true[:,:,1] / W
    yl = y_true[:,:,2] / H 
    xr = (y_true[:,:,3] + 1) / W                                          #    +1是为了包含右下边界。参考：4*4特征图时，切[0,0, 3,3]最终算出来是0-0.75的区域
    yr = (y_true[:,:,4] + 1) / H                                          #    +1是为了包含右下边界
    boxes = tf.stack([yl,xl, yr,xr], axis=-1)                             #    boxes参数的顺序是ymin,xmin, ymax,xmax，且必须为0~1之间
    binx = tf.expand_dims(tf.range(B, dtype=tf.int32), axis=0)
    binx = tf.repeat(binx, repeats=num, axis=1)
    binx = tf.squeeze(binx)
    boxes = tf.reshape(boxes, shape=(B*num, 4))
    #    TODO 疑问：既然是扣特征图，用这种方法直接扣下来不是更好吗，干嘛还要先扣2倍大小，再maxpooling
    crops = tf.image.crop_and_resize(fmaps, boxes=boxes, box_indices=binx, crop_size=crop_size)
    
    #    跑一次max_pooling，最终得到roipooling_ksize大小的特征图
    crops = tf.nn.max_pool(crops, ksize=[2,2], strides=2, padding='VALID')
    
    #    将[B*num, roipooling_ksize[0], roipooling_ksize[1], C]的结果还原为[B*num, roipooling_ksize[0], roipooling_ksize[1], C]
#     crops = tf.split(crops, num_or_size_splits=num_every_B, axis=0)
#     crops = tf.stack(crops, axis=0)
    return crops