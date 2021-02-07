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
import matplotlib.pyplot as plot
import numpy as np

import data.dataset as ds
import data.dataset_rois as ds_rois
import utils.conf as conf
import models.rpn as rpn
from models.layers.rpn.preprocess import preprocess_like_array

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)


model_conf_fpath = conf.RPN.get_save_weights_dir() + '/conf_rpn_resnet34.yml'
model_fpath = conf.RPN.get_save_weights_dir() + '/rpn_resnet34_20_30.41.h5'

#    加载当时训练的配置
_, _, M_ROIS, M_RPN, M_CNNS, M_CTX, M_PROPOSALE, M_fast_rcnn = conf.load_conf_yaml(model_conf_fpath)


#    初始化RPN网络
rpn_model = rpn.RPNModel(cnns_name=M_RPN.get_cnns(), 
                         learning_rate=M_RPN.get_train_learning_rate(),
                         scaling=M_CNNS.get_feature_map_scaling(), 
                         K=M_ROIS.get_K(),
                         cnns_base_channel_num=M_CNNS.get_base_channel_num(),
                         train_cnns=True,
                         train_rpn=True,
                         loss_lamda=M_RPN.get_loss_lamda(),
                         is_build=True)
rpn_model.load_model_weight(model_fpath)
#    设置cnns不参与训练
rpn_model.cnns.trainable = False
rpn_model.rpn.trainable = False
rpn_model.show_info()


################################################################################################3
#    验证分类准确率和回归MAE
################################################################################################3
count = conf.DATASET.get_count_test()
count = 10
X_test, Y_test = ds_rois.rpn_test_db(image_dir=conf.DATASET.get_in_test(), 
                                     count=count, 
                                     rois_out=conf.ROIS.get_test_rois_out(), 
                                     is_rois_mutiple_file=conf.DATASET.get_label_test_mutiple(), 
                                     count_positives=conf.ROIS.get_positives_every_image(), 
                                     count_negative=conf.ROIS.get_negative_every_image(), 
                                     x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                     y_preprocess=lambda y:preprocess_like_array(y))
fmaps = rpn_model.test(X_test, batch_size=conf.ROIS.get_batch_size())
TP, TN, FP, FN, P, N = rpn_model.test_cls(fmaps, Y_test)
print('分类准确率((TP + TN) / (P + N)):', (TP + TN) / (P + N))
mae = rpn_model.test_reg(fmaps, Y_test)
print('回归平均绝对误差(MAE):', mae)
 
 
 
################################################################################################
#    验证模型生成建议框
################################################################################################
#    准备测试集
count = 5
X, Y = ds.load_XY_np(count=count,
                      image_dir=conf.DATASET.get_in_test(), 
                      label_fpath=conf.DATASET.get_label_test(), 
                      is_label_mutiple_file=conf.DATASET.get_label_test_mutiple(), 
                      x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                      y_preprocess=None)

  
#    拿到测试数据全部的建议框
fmaps = rpn_model.test(X, batch_size=conf.ROIS.get_batch_size())
anchors = rpn_model.candidate_box_from_fmap(fmaps=fmaps, 
                                            threshold_prob=0.9, 
                                            threshold_iou=0.9)
  
#    在图上划出候选框与标签
def show_anchors_labels(X, anchors, labels, show_labels=True, show_anchors=True):
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
         
    #    绘制labels
    if (show_labels):
        idx_label = 0
        for a in labels:
            idx_label += 1
            if (idx_label == 1): continue
            x,y, w, h = a[1], a[2], a[3], a[4]
            #    按比例缩放
            rect = plot.Rectangle((x,y), w, h, fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(rect)
            pass
        pass
         
    #    绘制anchors
    if (show_anchors):
        print('anchors.count:', len(anchors))
        for a in anchors:
            print((a[1], a[2]), math.fabs(a[3] - a[1]), math.fabs(a[4] - a[2]))
            rect = plot.Rectangle((a[1], a[2]), math.fabs(a[3] - a[1]), math.fabs(a[4] - a[2]), fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            pass
        pass
         
    #    绘制图片
    X = X / 2. + 0.5
    plot.axis("off")
    plot.imshow(X)
    plot.show()
    pass

idx = 1
show_anchors_labels(X[idx], anchors[idx], Y[idx], show_labels=True, show_anchors=True)

