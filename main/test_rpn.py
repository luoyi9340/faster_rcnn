# -*- coding: utf-8 -*-  
'''
测试rpn

@author: luoyi
Created on 2021年1月8日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import math
import numpy as np
import matplotlib.pyplot as plot

import data.dataset as ds
import models.rpn as rpn
import utils.conf as conf


#    初始化RPN网络
rpn_model = rpn.RPNModel(cnns_name=conf.RPN.get_cnns(), 
                         learning_rate=conf.RPN.get_train_learning_rate(),
                         scaling=conf.CNNS.get_feature_map_scaling(), 
                         train_cnns=True,
                         train_rpn=True,
                         loss_lamda=conf.RPN.get_loss_lamda(),
                         is_build=True)
rpn_model.load_model_weight(conf.RPN.get_save_weights_dir() + '/rpn_resnet34.h5')


#    准备测试集
X, Y = ds.load_XY_np(count=conf.DATASET.get_count_test(), 
                      image_dir=conf.DATASET.get_in_test(), 
                      label_fpath=conf.DATASET.get_label_test(), 
                      is_label_mutiple_file=conf.DATASET.get_label_test_mutiple(), 
                      y_preprocess=None)

#    拿到测试数据全部的建议框
fmaps = rpn_model.test(X, batch_size=conf.RPN.get_train_batch_size())
anchors = rpn_model.candidate_box_from_fmap(fmaps=fmaps, 
                                            threshold_prob=conf.RPN.get_nms_threshold_positives(), 
                                            threshold_iou=conf.RPN.get_nms_threshold_iou())


#    在图上划出候选框与标签
def show_anchors_labels(X, anchors, labels):
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    
    #    绘制anchors
    print('anchors.count:', len(anchors))
    for a in anchors:
        rect = plot.Rectangle((a[1], a[2]), math.fabs(a[3] - a[1]), math.fabs(a[4] - a[2]), fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        pass
    
    #    绘制labels
    idx_label = 0
    for a in labels:
        idx_label += 1
        if (idx_label == 1): continue
        x,y, w, h = a[1], a[2], a[3], a[4]
        #    按比例缩放
        rect = plot.Rectangle((x,y), w, h, fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)
        pass
    
    #    绘制图片
#     X = ((X / 2) + 0.5) * 255.
    plot.imshow(X)
    plot.show()
    pass


show_anchors_labels(X[0], anchors[0], Y[0])

