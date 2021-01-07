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
                          count_positives=conf.RPN.get_train_positives_every_image(),
                          count_negative=conf.RPN.get_train_negative_every_image()):
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
                            (h, w, 2, k)代表(h,w)点的第k个anchor的t[y]偏移比
                            (h, w, 2, k)代表(h,w)点的第k个anchor的t[w]缩放比
                            (h, w, 2, k)代表(h,w)点的第k个anchor的t[h]缩放比
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
    y_true_positives = y_true_positives[y_true_positives[:,0] > 0]
    y_true_negative = y_true_negative[y_true_negative[:,0] > 0]
    
    #    写入正负样本的前背景概率
    #    正样本的(h, w, 0, k)置为1
    idx_fmap = y_true_positives[:, [5,6]].astype(np.int8)
    idx_anchor = y_true_positives[:, [7,8]].astype(np.int8)
    idx_anchor = idx_anchor[:,0] * len(conf.RPN.get_roi_scales()) + idx_anchor[:,1]
    idx_anchor = tuple(idx_anchor)
    fmap_w = tuple(idx_fmap[:,0])
    fmap_h = tuple(idx_fmap[:,1])
    zero_template[fmap_h, fmap_w, 0, idx_anchor] = 1            #    (h, w, 0, k)为t前景概率
    #    负样本的(h, w, 1, k)置为-1
    idx_fmap = y_true_negative[:, [5,6]].astype(np.int8)
    idx_anchor = y_true_negative[:, [7,8]].astype(np.int8)
    idx_anchor = idx_anchor[:,0] * len(conf.RPN.get_roi_scales()) + idx_anchor[:,1]
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
    idx_anchor = idx_anchor[:,0] * len(conf.RPN.get_roi_scales()) + idx_anchor[:,1]
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
                    (分类正样本的特征图(batch_size, h, w, 2, K)，
                     分类负样本的特征图(batch_size, h, w, 2, K)，
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
