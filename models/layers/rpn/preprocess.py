# -*- coding: utf-8 -*-  
'''
标签预处理

@author: luoyi
Created on 2021年1月6日
'''
import numpy as np
import tensorflow as tf

import utils.conf as conf


#    将标签数据整形为feature_maps一致的形状
def preprocess_like_fmaps(y_true, 
                          shape=(12, 30, 6, 15), 
                          count_positives=conf.ROIS.get_positives_every_image(),
                          count_negative=conf.ROIS.get_negative_every_image()):
    '''自定义的loss和metrics无法直接用numpy。这里做类似one_hot的事情，提前将y_true数据整形成y_pred一致。loss和metrics只做简单的事情
        @param y_true: numpy:(count_positives+count_negative, 14)
                        [[IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x, y, w, h]
                         ...前count_positives个为正样本...
                         ...后count_negative个为负样本...]
        @param shape: y_pred的输出形状。RPNNet.get_output_shape()
                        (h, w, 2, K)代表cls结果
                            (h, w, 0, k)代表(h,w)点的第k个anchor的前景概率
                            (h, w, 1, k)代表(h,w)点的第k个anchor的背景概率
                        (h, w, 4, K)代表reg结果
                            (h, w, 2, k)代表(h,w)点的第k个anchor的t[x]偏移比
                            (h, w, 3, k)代表(h,w)点的第k个anchor的t[y]偏移比
                            (h, w, 4, k)代表(h,w)点的第k个anchor的t[w]缩放比
                            (h, w, 5, k)代表(h,w)点的第k个anchor的t[h]缩放比
        @param count_positives: y_true中前count_positives个为正样本
        @param count_negative: y_true中后count_negative个负样本
        @return: 与shape形状一致的numpy
    '''
    #    全0模板
    zero_template = np.zeros(shape=shape)
    #    切分正负样本
    y_true_positives = y_true[:count_positives, :]
    y_true_negative = y_true[count_positives:, :]
    #    过滤掉IoU<0的
    y_true_positives = y_true_positives[y_true_positives[:,0] >= 0]
    y_true_negative = y_true_negative[y_true_negative[:,0] >= 0]            #    还真能碰到IoU=0的数据
    
    #    写入正负样本的前背景概率
    #    正样本的(h, w, 0, k)置为1
    idx_fmap = y_true_positives[:, [5,6]].astype(np.int8)
    idx_anchor = y_true_positives[:, [7,8]].astype(np.int8)
    idx_anchor = idx_anchor[:,0] * len(conf.ROIS.get_roi_scales()) + idx_anchor[:,1]
    idx_anchor = tuple(idx_anchor)
    fmap_w = tuple(idx_fmap[:,0])
    fmap_h = tuple(idx_fmap[:,1])
    zero_template[fmap_h, fmap_w, 0, idx_anchor] = 1            #    (h, w, 0, k)为t前景概率
    #    负样本的(h, w, 1, k)置为-1
    idx_fmap = y_true_negative[:, [5,6]].astype(np.int8)
    idx_anchor = y_true_negative[:, [7,8]].astype(np.int8)
    idx_anchor = idx_anchor[:,0] * len(conf.ROIS.get_roi_scales()) + idx_anchor[:,1]
    idx_anchor = tuple(idx_anchor)
    fmap_w = tuple(idx_fmap[:,0])
    fmap_h = tuple(idx_fmap[:,1])
    zero_template[fmap_h, fmap_w, 1, idx_anchor] = -1            #    (h, w, 1, k)为背景概率
    
    #    写入正样本的T[*]
    Gx = y_true_positives[:,10] + y_true_positives[:,12]/2      #    y_true中标签的x是左上坐标，这里换算下
    Gy = y_true_positives[:,11] + y_true_positives[:,13]/2      #    y_true中标签的y是左上坐标，这里换算下
    Gw = y_true_positives[:,12]
    Gh = y_true_positives[:,13]
    Px = y_true_positives[:,1]
    Py = y_true_positives[:,2]
    Pw = y_true_positives[:,3]
    Ph = y_true_positives[:,4]
    
    Tx = (Gx - Px) * Pw                                         #    计算t[x] = (G[x] - P[x]) * P[w]
    Ty = (Gy - Py) * Ph                                         #    计算t[y] = (G[y] - P[y]) * P[h]
    Tw = np.log(Gw / Pw)                                        #    计算t[w] = log(G[w] / P[w])
    Th = np.log(Gh / Ph)                                        #    计算t[h] = log(G[h] / P[h])
    #    根据fmap索引和anchor索引写入模板
    idx_fmap = y_true_positives[:, [5,6]].astype(np.int8)
    idx_anchor = y_true_positives[:, [7,8]].astype(np.int8)
    idx_anchor = idx_anchor[:,0] * len(conf.ROIS.get_roi_scales()) + idx_anchor[:,1]
    idx_anchor = tuple(idx_anchor)
    fmap_w = tuple(idx_fmap[:,0])
    fmap_h = tuple(idx_fmap[:,1])
    zero_template[fmap_h, fmap_w, 2, idx_anchor] = Tx           #    (h, w, 2, k)为t[x]
    zero_template[fmap_h, fmap_w, 3, idx_anchor] = Ty           #    (h, w, 3, k)为t[y]
    zero_template[fmap_h, fmap_w, 4, idx_anchor] = Tw           #    (h, w, 4, k)为t[w]
    zero_template[fmap_h, fmap_w, 5, idx_anchor] = Th           #    (h, w, 5, k)为t[h]
    
    ymaps = zero_template
    #    验算：(h, w, 2:, K)>0的数量应该是(h, w, 0, K)>0的数量的4倍
    #    取(h, w, 0, K)>0的数量（也不一定，会存在anchor的偏移比==0的情况，此时Tx或Ty算出来正好==0）
#     count_p_cls = np.count_nonzero(ymaps[:,:,0,:])
#     count_p_reg = np.count_nonzero(ymaps[:,:,2:,:])
#     print(count_p_cls, count_p_reg)
    return ymaps

#    将标签数据整形为需要用到的数据
def preprocess_like_array(y_true):
    '''将标签数据整形为需要用到的数据
        @param y_true: numpy (positives_every_image+negative_every_image, 14)
                        [
                            [IoU得分, anchor原图坐标(中心点)宽高, anchor特征图坐标(中心点), anchor面积比宽高比索引, label分类值, label原图坐标(左上点)宽高]
                            [IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x(label左上点), y(label左上点), w, h]
                            ...
                        ]
        @param is_p: 是否正样本
        @return: [IoU得分, idx_w, idx_h, idx_area, idx_scales, 正负样本区分(1 | -1), t[x], t[y], t[w], t[h]]
    '''
    Gx = y_true[:,10] + y_true[:,12]/2
    Gy = y_true[:,11] + y_true[:,13]/2
    Gw = y_true[:,12]
    Gh = y_true[:,13]
    Px = y_true[:,1]
    Py = y_true[:,2]
    Pw = y_true[:,3]
    Ph = y_true[:,4]
    #    对于负样本，不需要计算t*
    idx_p = y_true[:,9] >= 0
    zero_tmp = np.zeros_like(Gx)
    tx = np.where(idx_p, (Gx - Px) * Pw, zero_tmp)                      #    t[x] = (G[x] - P[x]) * P[w]
    ty = np.where(idx_p, (Gy - Py) * Ph, zero_tmp)                      #    t[y] = (G[y] - P[y]) * P[h]
    tw = np.where(idx_p, np.log(Gw / Pw) , zero_tmp)                     #    t[w] = log(G[w] / P[w])
    th = np.where(idx_p, np.log(Gh / Ph) , zero_tmp)                     #    t[h] = log(G[h] / P[h])
    #    计算正负样本标记
    one_tmp = np.ones_like(Gx)
    minus_tmp = -one_tmp
    pn_tag = np.where(idx_p, one_tmp, minus_tmp)
    y_true = np.vstack([y_true[:,0], y_true[:,5], y_true[:,6], y_true[:,7], y_true[:,8], pn_tag, tx, ty, tw, th]).T
    return y_true



#    通过ytrue_maps取分类的正负样本/回归的正样本
def takeout_sample(ytrue_maps, feature_maps):
    '''通过ytrue_maps从feature_maps中取分类的正负样本，回归的正样本
            ytrue_maps中分类的正样本=1，负样本=-1
            ytrue_maps中回归的正样本>0，没有负样本
            其余都是0
        @param ytrue_maps: 通过preprocess_like_fmaps生成的与feature_maps形状一致的特征图。外面加上batch_size维度
                                (batch_size, h, w, 6, K)
        @param feature_maps: cnns + rpn输出的特征图
                                (batch_size, h, w, 6, K)
        return (fmaps_cls_p, fmaps_cls_n, fmaps_reg_p), (ymaps_cls_p, ymaps_cls_n, ymaps_reg_p)
                    (分类正样本的特征图(batch_size, h, w, 0, K)，
                     分类负样本的特征图(batch_size, h, w, 1, K)，
                     回归正样本的特征图(batch_size, h, w, 4, K))
    '''
    #    从6=2+4中切分出分类 / 回归的特征图
    [ytrue_maps_cls, ytrue_maps_reg] = tf.split(ytrue_maps, [2, 4], axis=3)
    [fmaps_cls, fmaps_reg] = tf.split(feature_maps, [2, 4], axis=3)
    
    zeros_tmp_cls = tf.zeros_like(ytrue_maps_cls)
    zeros_tmp_reg = tf.zeros_like(ytrue_maps_reg)
    #    按照cls中正样本=1，负样本=-1，reg中正样本>0，没有负样本的规则摘出感兴趣的样本
    fmaps_cls_p = tf.where(ytrue_maps_cls > 0, fmaps_cls, zeros_tmp_cls)
    fmaps_cls_n = tf.where(ytrue_maps_cls < 0, fmaps_cls, zeros_tmp_cls)
    fmaps_reg_p = tf.where(ytrue_maps_reg > 0, fmaps_reg, zeros_tmp_reg)  
    ymaps_cls_p = tf.where(ytrue_maps_cls > 0, ytrue_maps_cls, zeros_tmp_cls)
    ymaps_cls_n = tf.where(ytrue_maps_cls < 0, ytrue_maps_cls, zeros_tmp_cls)
    ymaps_reg_p = tf.where(ytrue_maps_reg > 0, ytrue_maps_reg, zeros_tmp_reg)
    
    return (fmaps_cls_p, fmaps_cls_n, fmaps_reg_p), (ymaps_cls_p, ymaps_cls_n, ymaps_reg_p)
#    通过ytrue_maps取分类的正负样本/回归的正样本
def takeout_sample_np(ytrue_maps, feature_maps):
    '''通过ytrue_maps从feature_maps中取分类的正负样本，回归的正样本
            ytrue_maps中分类的正样本=1，负样本=-1
            ytrue_maps中回归的正样本>0，没有负样本
            其余都是0
        @param ytrue_maps: 通过preprocess_like_fmaps生成的与feature_maps形状一致的特征图。外面加上batch_size维度
                                (batch_size, h, w, 6, K)
        @param feature_maps: cnns + rpn输出的特征图
                                (batch_size, h, w, 6, K)
        return (fmaps_cls_p, fmaps_cls_n, fmaps_reg_p), (ymaps_cls_p, ymaps_cls_n, ymaps_reg_p)
                    (分类正样本的特征图(batch_size, h, w, 2, K)，
                     分类负样本的特征图(batch_size, h, w, 2, K)，
                     回归正样本的特征图(batch_size, h, w, 4, K))
    '''
    #    从6=2+4中切分出分类 / 回归的特征图
    [ytrue_maps_cls, ytrue_maps_reg] = np.split(ytrue_maps, [2], axis=3)
    [fmaps_cls, fmaps_reg] = np.split(feature_maps, [2], axis=3)
    
    zeros_tmp_cls = np.zeros_like(ytrue_maps_cls)
    zeros_tmp_reg = np.zeros_like(ytrue_maps_reg)
    #    按照cls中正样本=1，负样本=-1，reg中正样本>0，没有负样本的规则摘出感兴趣的样本
    fmaps_cls_p = np.where(ytrue_maps_cls > 0, fmaps_cls, zeros_tmp_cls)
    fmaps_cls_n = np.where(ytrue_maps_cls < 0, fmaps_cls, zeros_tmp_cls)
    fmaps_reg_p = np.where(ytrue_maps_reg > 0, fmaps_reg, zeros_tmp_reg)  
    ymaps_cls_p = np.where(ytrue_maps_cls > 0, ytrue_maps_cls, zeros_tmp_cls)
    ymaps_cls_n = np.where(ytrue_maps_cls < 0, ytrue_maps_cls, zeros_tmp_cls)
    ymaps_reg_p = np.where(ytrue_maps_reg > 0, ytrue_maps_reg, zeros_tmp_reg)
    
    return (fmaps_cls_p, fmaps_cls_n, fmaps_reg_p), (ymaps_cls_p, ymaps_cls_n, ymaps_reg_p)

#    根据y_true从fmaps中取全部的样本
def takeout_sample_array(y_true, fmaps, roi_areas=conf.ROIS.get_roi_areas(), roi_scales=conf.ROIS.get_roi_scales()):
    '''根据y_true从fmaps中取全部的样本
        @param y_true: Tensor(batch_size, num, 10)
                        [IoU得分(正样本>0，负样本<0), idx_w, idx_h, idx_area, idx_scales, 正负样本区分(1 | -1), t[x], t[y], t[w], t[h]]
        @param fmaps: Tensor(batch_size, H, W, 6, K)
                        cnns得到的特征图
                        (batch_size, H, W, 0, K) 正样本特征图
                        (batch_size, H, W, 1, K) 负样本特征图
                        (batch_size, H, W, 4, K) 正样本回归特征图
                            (batch_size, H, W, 2, K) 正样本d[x]
                            (batch_size, H, W, 3, K) 正样本d[y]
                            (batch_size, H, W, 4, K) 正样本d[w]
                            (batch_size, H, W, 5, K) 正样本d[h]
        @param roi_areas: 生成anchor时的面积比，用于通过idx_area, idx_scales还原第几个k
        @param roi_scales: 生成anchor时的长宽比，用于通过idx_area, idx_scales还原第几个k
        @return anchors
                    Tensor(batch_size, num, 7) 
                        最后7维解释：[±1, 正样本概率，负样本概率，d[x], d[y], d[w], d[h]]
                                    ±1标识是正样本还是负样本
                                    与y_true中的每行一一对应
    '''
    #    维度转换(batch_size, H, W, 6, K) -> (batch_size, H, W, K, 6) 每行的6个数据代表[正样本概率，负样本概率，d[x], d[y], d[w], d[h]]
    fmaps = tf.transpose(fmaps, perm=[0,1,2,4,3])
    #    根据y_true从fmaps中生成anchors[batch_size=None, num=正负样本数, 7]
    num_every_b = tf.math.count_nonzero(y_true[:,:,0], axis=1)                  #    每个batch的样本数。其实不用取，每个batch的样本数应该都是每张图的正负样本数之和
    num_b = tf.math.count_nonzero(num_every_b)                                  #    总batch数
    idx_ = tf.range(num_b, dtype=tf.int32)
    idx_ = tf.repeat(tf.expand_dims(idx_, axis=-1), repeats=num_every_b[0], axis=1)
    idx_w = tf.cast(y_true[:,:, 1], dtype=tf.int32)
    idx_h = tf.cast(y_true[:,:, 2], dtype=tf.int32)
    idx_k = tf.cast(y_true[:,:, 3] * len(roi_scales) + y_true[:,:, 4], dtype=tf.int32)
    np_tag = tf.cast(y_true[:,:, 5], dtype=tf.float32)
    idx = tf.stack([idx_, idx_h, idx_w, idx_k], axis=2)
    anchors = tf.gather_nd(fmaps, idx)    
    anchors = tf.concat([tf.expand_dims(np_tag, axis=-1), anchors], axis=2)
    
    return anchors


#    从fmaps拿全部判定为正样本的anchor
def all_positives_from_fmaps(fmaps, 
                             threshold=conf.RPN.get_nms_threshold_positives(), 
                             K=conf.ROIS.get_K(),
                             feature_map_scaling=conf.CNNS.get_feature_map_scaling(),
                             roi_areas=conf.ROIS.get_roi_areas(),
                             roi_scales=conf.ROIS.get_roi_scales()):
    '''从fmaps拿全部判定为正样本的anchor
        @param fmaps: numpy(num, H, W, 6, K) RPNModel拿到的特征图
        @param threshold: 判定正样本的阈值。(num, H, W, 0, K) > 此值判定为正样本
        @param feature_map_scaling: 特征图缩放比例
        @param roi_areas: anchor面积比
        @param roi_scales: anchor宽高比
        @return: [narray(num, 5)]      numpy数组
                    [正样本概率, xl,yl, xr,yr, ]
    '''
    B, H, W = fmaps.shape[0], fmaps.shape[1], fmaps.shape[2]
    roi_areas = np.array(roi_areas)
    roi_scales = np.array(roi_scales)
    
    [fmaps_p, _, fmaps_reg] = np.split(fmaps, [1,2], axis=3)
    fmaps_p[fmaps_p <= threshold] = 0
    
    #    把正样本概率<阈值的reg清零
    tmp = np.zeros_like(fmaps_p)
    tmp[fmaps_p > 0.5] = 1
    tmp = np.repeat(tmp, 4, axis=3)
    fmaps_reg = tmp * fmaps_reg
    
    fmaps = np.concatenate([fmaps_p, fmaps_reg], axis=3)            #    a.shape=(batch_size, h, w, 5, K)。a[b,h,w]的每列代表一个anchor，每列数据为[prob, d[x], d[y], d[w], d[h]]
    fmaps = np.transpose(fmaps, axes=(0,1,2, 4,3))                  #    a.shape=(batch_size, h, w, K, 5)。a[b,h,w]的每行代表一个anchor，每行数据为[prob, d[x], d[y], d[w], d[h]]
    
    #    给a的每行追加h[0,H), w[0,W), anchor[0,K)索引。扩充后a.shape=(batch_size, h, w, K, 8)。a[b,h,w]每行数据为[prob, d[x], d[y], d[w], d[h], idx_h, idx_w, idx_anchor]
    idx_anchor = np.arange(K)                                       #    得到单位anchor索引shape=(K)
    idx_anchor = np.concatenate([idx_anchor for _ in range(H*W)])
    idx_anchor = np.reshape(idx_anchor, newshape=(len(idx_anchor), 1))
    idx_w, idx_h = np.meshgrid(range(W), range(H))
    idx_w = np.reshape(idx_w, newshape=H*W)
    idx_h = np.reshape(idx_h, newshape=H*W)
    idx = np.concatenate([np.expand_dims(idx_h, axis=0), np.expand_dims(idx_w, axis=0)], axis=0)
    idx = np.transpose(idx, axes=(1, 0))                            #    得到单位h,w索引 shape=(H*W, 2)
    idx = np.repeat(idx, K, axis=0)                                 #    (H*W, 2)扩展为(H*W*K, 2)
    idx = np.concatenate([idx, idx_anchor], axis=1)
    idx = np.reshape(idx, newshape=(H, W, K, 3))
    idx = np.repeat(np.expand_dims(idx, axis=0), B, axis=0)
    fmaps = np.concatenate([fmaps, idx], axis=4)                    #    追加索引, fmaps.shape(batch_size, h, w, K, 8)
    
    #    根据idx_h,idx_w(相对于特征图), idx_anchor 和 d[x],d[y],d[w],d[h] 换算出原图坐标
    #    特征图每个像素点对应原图中心点坐标 = (特征图坐标x * 缩放比例 + 缩放比例/2, 特征图坐标y * 缩放比例 + 缩放比例/2)
    xc, yc = fmaps[:, :,:, :, 6] * feature_map_scaling + feature_map_scaling/2, fmaps[:, :,:, :, 5].astype(np.int8) * feature_map_scaling + feature_map_scaling/2
    #    根据idx_anchor计算每个anchor的宽高
    idx_areas = (fmaps[:, :,:, :, 7] / len(roi_scales)).astype(np.int8)
    idx_scales = (fmaps[:, :,:, :, 7] % len(roi_scales)).astype(np.int8)
    w = np.round(roi_areas[idx_areas] * roi_scales[idx_scales]).astype(np.float64)
    h = np.round(roi_areas[idx_areas] / roi_scales[idx_scales]).astype(np.float64)
    #    根据d[x], d[y], d[w], d[h]换算中心坐标和宽高
    xc = fmaps[:, :,:, :, 1] / w + xc                                           #    G_[x] = d[x]/P[w] + P[x]
    yc = fmaps[:, :,:, :, 2] / h + yc                                           #    G_[y] = d[y]/P[h] + P[y]
    w = np.exp(fmaps[:, :,:, :, 3]).astype(np.float64) * w                      #    G_[w] = exp(d[w]) * P[w]
    h = np.exp(fmaps[:, :,:, :, 4]).astype(np.float64) * h                      #    G_[h] = exp(d[h]) * P[h]
    xl, yl = (xc - w/2).astype(np.int16), (yc - h/2).astype(np.int16)                                     #    左上点坐标 = (xc - w/2, yc - h/2)
    xr, yr = (xc + w/2).astype(np.int16), (yc + h/2).astype(np.int16)                                     #    右下点坐标 = (xc + w/2, yc + h/2)
    #    拼接后anchors.shape=(batch_size, h, w, k, 5)
    anchors = np.concatenate([np.expand_dims(fmaps[:, :,:, :, 0], axis=-1), 
                              np.expand_dims(xl, axis=-1), 
                              np.expand_dims(yl, axis=-1), 
                              np.expand_dims(xr, axis=-1), 
                              np.expand_dims(yr, axis=-1)], axis=-1).astype(np.float32)
    
    #    过滤掉0>xl>IMAGE_WEIGHT, 0>yl>IMAGE_HEIGHT, 0>xr>IMAGE)WEIGHT, 0>yr>IMAGE_WEIGHT的部分（将概率置为-1，下面会统一过滤掉）
    anchors[(anchors[:, :,:, :, 1] < 0) + (anchors[:, :,:, :, 1] > conf.IMAGE_WEIGHT)] = [-1, 0,0,0,0]
    anchors[(anchors[:, :,:, :, 2] < 0) + (anchors[:, :,:, :, 2] > conf.IMAGE_HEIGHT)] = [-1, 0,0,0,0]
    anchors[(anchors[:, :,:, :, 3] < 0) + (anchors[:, :,:, :, 3] > conf.IMAGE_WEIGHT)] = [-1, 0,0,0,0]
    anchors[(anchors[:, :,:, :, 4] < 0) + (anchors[:, :,:, :, 4] > conf.IMAGE_HEIGHT)] = [-1, 0,0,0,0]
    #    过滤掉概率小于阈值的。每张图片的候选框数量是不一样的，所以不能组成一个narray对象
    idxs = anchors[:, :,:, :, 0] >= threshold
    res = []
    for fmap, idx in zip(anchors, idxs):
        as_ = fmap[idx]
        res.append(as_)
        pass
    return res



