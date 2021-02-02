# -*- coding: utf-8 -*-  
'''
tensorflow 测试

@author: luoyi
Created on 2021年1月1日
'''
import tensorflow as tf
import numpy as np

import utils.math_expand as me

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)  


#    roi pooling
B, H, W, C = 2, 10, 10, 1
fmaps = tf.reshape(tf.range(B * H * W * C), shape=(B, H, W, C))
y_true = tf.convert_to_tensor([
                                [
                                    [0, 0,0,9,9, 1,1,1,1],
                                    [0, 5,5,9,9, 1,1,1,1],
                                ],
                                [
                                    [0, 0,0,5,5, 1,1,1,1],
                                    [0, 5,5,9,9, 1,1,1,1],
                                ]
                              ], dtype=tf.float32)
num = y_true.shape[1]
print(tf.squeeze(fmaps))

def roi_pooling(fmaps, y_true, roipooling_ksize=[5, 5], op='max'):
    #    取所有的左上/右下坐标
    xl = tf.cast(tf.math.floor(y_true[:,:,1]), tf.int32)
    yl = tf.cast(tf.math.floor(y_true[:,:,2]), tf.int32)
    xr = tf.cast(tf.math.ceil(y_true[:,:,3]), tf.int32)
    yr = tf.cast(tf.math.ceil(y_true[:,:,4]), tf.int32)
    rects = tf.stack([xl,yl, tf.abs(xr - xl),tf.abs(yr - yl)], axis=2)
    rects = tf.reshape(rects, shape=(rects.shape[0] * rects.shape[1], rects.shape[2]))
    
    proposal_fmaps = []
    idx_fmaps = 0
    idx = 0
    crt_fmaps = fmaps[0]
    crt_fmaps = tf.expand_dims(crt_fmaps, axis=0)
    for (x,y, w,h) in rects:
        if (idx >= num):
            idx_fmaps += 1
            idx = 0
            crt_fmaps = fmaps[idx_fmaps]
            crt_fmaps = tf.expand_dims(crt_fmaps, axis=0)
            pass
        else:
            idx += 1
            pass
        
        proposal = tf.image.crop_to_bounding_box(image=crt_fmaps, 
                                                 offset_height=x, 
                                                 offset_width=y, 
                                                 target_height=h, 
                                                 target_width=w)
        
        r = roi_pooling_proposal(proposal, roipooling_ksize, op)
        proposal_fmaps.append(r)
        pass
    
    return tf.concat(proposal_fmaps, axis=0)

def roi_pooling_proposal(proposal, roipooling_ksize=[5, 5], op='max'):
    H, W = proposal.shape[1], proposal.shape[2]
    #    如果宽高都出现不够切的情况，填充到[H, W]直接返回
    if (H < roipooling_ksize[0] and W < roipooling_ksize[1]):
        pad_h = roipooling_ksize[0] - H
        pad_w = roipooling_ksize[1] - W
        return tf.pad(proposal, paddings=[[0, 0], 
                                          [round(pad_h/2), pad_h - round(pad_h/2)], 
                                          [round(pad_w/2), pad_w - round(pad_w/2)], 
                                          [0, 0]])
    #    如果宽高中只有1方存在不够切的情况，填充到刚好够切1，再做roi_pooling
    if (H < roipooling_ksize[0]):
        pad_h = roipooling_ksize[0] - H
        proposal = tf.pad(proposal, paddings=[[0, 0], 
                                              [round(pad_h/2), pad_h - round(pad_h/2)], 
                                              [0, 0], 
                                              [0, 0]])
        H = proposal.shape[1]
        pass
    if (W < roipooling_ksize[0]):
        pad_w = roipooling_ksize[1] - W
        proposal = tf.pad(proposal, paddings=[[0, 0], 
                                              [0, 0], 
                                              [round(pad_w/2), pad_w - round(pad_w/2)], 
                                              [0, 0]])
        W = proposal.shape[2]
        pass
    
    hs = me.fairly_equalize(tag=H, num=roipooling_ksize[0])
    ws = me.fairly_equalize(tag=W, num=roipooling_ksize[1])
    #    先切H轴
    hsl = tf.split(proposal, num_or_size_splits=hs, axis=1)
    hsl_t = []
    for h in hsl:
        #    每个h切y轴
        wsl = tf.split(h, num_or_size_splits=ws, axis=2)
        wsl_t = []
        for w in wsl:
            if (op.lower() == 'avg'): vw = tf.nn.avg_pool(w, ksize=w.shape, strides=1, padding='VALID')
            else: vw = tf.nn.max_pool(w, ksize=[w.shape[1], w.shape[2]], strides=1, padding='VALID')
            wsl_t.append(vw)
            pass
        #    合并每个w轴的结果
        vh = tf.concat(wsl_t, axis=2)
        hsl_t.append(vh)
        pass
    #    合并每个h轴结果
    return tf.concat(hsl_t, axis=1)

pf = roi_pooling(fmaps, y_true)
print(tf.squeeze(pf))