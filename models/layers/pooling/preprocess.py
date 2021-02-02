# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2021年1月28日
'''
import tensorflow as tf
import utils.math_expand as me


#    roi_align
def roi_align(fmaps, y_true, roipooling_ksize=[7, 7]):
    '''roi align
        对选定的区域取双线性插值做reshape
        @param fmaps: tensor(batch_size, H, W, C)
        @param y_true: tensor(batch_size, num, 9)
                        [
                            [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                            [vidx, xl,yl,xr,yr, tx,th,tw,th]
                        ]
        @return: tensor(batch_size * num, roipooling_ksize[0], roipooling_ksize[1], C)
    '''
    #    取必要的参数
    H, W, _ = fmaps.shape[1], fmaps.shape[2], fmaps.shape[3]                #    特征图宽高
    B, num = y_true.shape[0], y_true.shape[1]                               #    batch_size, 每个batch_size中的proposal数
    crop_size = (roipooling_ksize[0] * 2, roipooling_ksize[1] * 2)
#     crop_size = roipooling_ksize
    
    #    将特征图统一切成roipooling_ksize * 2的大小
    xl = y_true[:,:,1] / (W - 1)
    yl = y_true[:,:,2] / (H - 1) 
    xr = (y_true[:,:,3]) / (W - 1)                                          #    -1是为了让W,H从0开始
    yr = (y_true[:,:,4]) / (H - 1)                                          
    boxes = tf.stack([yl,xl, yr,xr], axis=-1)                               #    boxes参数的顺序是ymin,xmin, ymax,xmax，且必须为0~1之间
    binx = tf.expand_dims(tf.range(B, dtype=tf.int32), axis=0)
    binx = tf.repeat(binx, repeats=num, axis=1)
    binx = tf.squeeze(binx)
    boxes = tf.reshape(boxes, shape=(B*num, 4))
    #    抠图，并用双线性插值reshape
    crops = tf.image.crop_and_resize(fmaps, boxes=boxes, box_indices=binx, crop_size=crop_size)
    crops = tf.nn.avg_pool(crops, ksize=[2,2], strides=2, padding='VALID')
    return crops


#    roi pooling
def roi_pooling(fmaps, y_true, roipooling_ksize=[5, 5], op='max'):
    '''roi_pooling
        对选定的区域取双线性插值做reshape
        @param fmaps: tensor(batch_size, H, W, C)
        @param y_true: tensor(batch_size, num, 9)
                        [
                            [分类索引, proposal左上/右下点坐标(相对特征图), proposal偏移比/缩放比]
                            [vidx, xl,yl,xr,yr, tx,th,tw,th]
                        ]
        @return: tensor(batch_size * num, roipooling_ksize[0], roipooling_ksize[1], C)
    '''
    _, num = y_true.shape[0], y_true.shape[1]                               #    batch_size, 每个batch_size中的proposal数
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
                                                 offset_height=y, 
                                                 offset_width=x, 
                                                 target_height=h, 
                                                 target_width=w)
        
        r = roi_pooling_proposal(proposal, roipooling_ksize, op)
        proposal_fmaps.append(r)
        pass
    res_pooling = tf.concat(proposal_fmaps, axis=0)
    return res_pooling
#    单个proposal做roi_pooling
def roi_pooling_proposal(proposal, roipooling_ksize=[5, 5], op='max'):
    '''roi_pooling_proposal
        对选定的区域取双线性插值做reshape
        @param fmaps: tensor(1, H, W, C)
        @param y_true: tensor(9) 
                            [vidx, xl,yl,xr,yr, tx,th,tw,th]
        @return: tensor(1, roipooling_ksize[0], roipooling_ksize[1], C)
    '''
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
            ksize = [w.shape[1], w.shape[2]]
            #    avg 或 max(默认)
            if (op.lower() == 'avg'): vw = tf.nn.avg_pool(w, ksize=ksize, strides=1, padding='VALID')
            else: vw = tf.nn.max_pool(w, ksize=ksize, strides=1, padding='VALID')
            wsl_t.append(vw)
            pass
        #    合并每个w轴的结果
        vh = tf.concat(wsl_t, axis=2)
        hsl_t.append(vh)
        pass
    #    合并每个h轴结果
    res_pooling = tf.concat(hsl_t, axis=1)
    return res_pooling

