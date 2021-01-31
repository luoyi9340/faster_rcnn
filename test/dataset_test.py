# -*- coding: utf-8 -*-  
'''
训练loss中有数据集总数不对的情况，怀疑数据还是有问题。。。
这里主要测试：
    1 数据集生成过程y_preprocess
    2 y_true取anchors过程

@author: luoyi
Created on 2021年1月22日
'''
import tensorflow as tf
import numpy as np
import math
import collections as collections
import matplotlib.pyplot as plot

import utils.conf as conf
import data.dataset_rois as ds_rois
import data.dataset_proposals as ds_proposals
from models.layers.fast_rcnn.preprocess import preprocess_like_array
from models.layers.pooling.preprocess import roi_align


#    验算proposal。从json读出来的proposal在原图上打出来看哈
batch_size = conf.PROPOSALES.get_batch_size()
epochs = conf.PROPOSALES.get_epochs()
total_samples = ds_proposals.total_samples(proposal_out=conf.PROPOSALES.get_train_proposal_out(), 
                                           is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                           count=conf.DATASET.get_count_train(), 
                                           proposal_every_image=conf.PROPOSALES.get_proposal_every_image())
train_count = conf.DATASET.get_count_train()
db_train, train_y_crt_queue = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_train(), 
                                                                count=train_count, 
                                                                proposals_out=conf.PROPOSALES.get_train_proposal_out(), 
                                                                is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                                                proposal_every_image=conf.PROPOSALES.get_proposal_every_image(), 
                                                                batch_size=batch_size, 
                                                                epochs=epochs, 
                                                                shuffle_buffer_rate=conf.PROPOSALES.get_shuffle_buffer_rate(), 
                                                                ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9), 
                                                                x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                                                y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()),
#                                                                 y_preprocess=None
                                                                )

#    图片展示
#    图片展示
def show_img(X, proposales, is_show_proposales=True, is_show_labels=True, feature_map_scaling=conf.CNNS.get_feature_map_scaling()):
    X = X / 2. + 0.5
    
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    
    label_dict = {}
    for proposale in proposales:
        if (is_show_proposales):
            x = proposale[1] * feature_map_scaling
            y = proposale[2] * feature_map_scaling
            w = np.abs(proposale[3] * feature_map_scaling - proposale[1] * feature_map_scaling)
            h = np.abs(proposale[4] * feature_map_scaling - proposale[2] * feature_map_scaling)
            rect = plot.Rectangle((x, y), w, h, fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
            pass
        
        if (is_show_labels):
            if (label_dict.get(proposale.numpy()[0]) is None):
                label_dict[proposale.numpy()[0]] = 1
                  
                tx = proposale[5]
                ty = proposale[6]
                tw = proposale[7]
                th = proposale[8]
                Px = proposale[1] * feature_map_scaling
                Py = proposale[2] * feature_map_scaling
                Pw = np.abs(proposale[3] * feature_map_scaling - proposale[1] * feature_map_scaling)
                Ph = np.abs(proposale[4] * feature_map_scaling - proposale[2] * feature_map_scaling)
                x = tx / Pw + Px
                y = ty / Ph + Py
                w = math.exp(tw) * Pw
                h = math.exp(th) * Ph
                rect = plot.Rectangle((x, y), w, h, fill=False, edgecolor = 'blue',linewidth=1)
                ax.add_patch(rect)
                pass
            pass
        pass
    
    plot.imshow(X)
    plot.show()
    pass

#    打印长宽比看看
def show_wh(x, y):
#     w = np.abs(y[:,3] - y[:,1])
#     h = np.abs(y[:,4] - y[:,2])
#     print(*zip(w, h))
    pass


show_idx = 2
idx = 0
for x, y in db_train:
    if (idx < show_idx):
        idx += 1
        continue
    show_wh(y[0])
    show_wh(y[1])
#     show_img(x[0], y[0], is_show_proposales=False, is_show_labels=True)
    break
    pass


