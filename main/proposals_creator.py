# -*- coding: utf-8 -*-  
'''
生成建议框

@author: luoyi
Created on 2021年1月8日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import numpy as np

import data.dataset_proposals as ds_ppsl
import utils.conf as conf
import models.rpn as rpn

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)


model_conf_fpath = conf.RPN.get_save_weights_dir() + '/conf_rpn_resnet34.yml'
model_fpath = conf.RPN.get_save_weights_dir() + '/rpn_resnet34.h5'

#    加载当时训练的配置
_, _, M_ROIS, M_RPN, M_CNNS, M_CTX, M_PRPPOSAL = conf.load_conf_yaml(model_conf_fpath)


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
#    设置cnns和rpn不参与训练
rpn_model.cnns.trainable = False
rpn_model.rpn.trainable = False

rpn_model.show_info()


#    建议框生成器
proposals_creator = ds_ppsl.ProposalsCreator(threshold_nms_prob=conf.RPN.get_nms_threshold_positives(),
                                               threshold_nms_iou=conf.RPN.get_nms_threshold_iou(),
                                               proposal_iou=conf.PROPOSALES.get_proposal_iou(),
                                               proposal_every_image=conf.PROPOSALES.get_proposal_every_image(),
                                               rpn_model=rpn_model)

#    生成建议框
proposals_creator.create(proposals_out=conf.PROPOSALES.get_train_proposal_out(),
                         image_dir=conf.DATASET.get_in_train(), 
                         label_path=conf.DATASET.get_label_train(), 
                         is_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                         count=conf.DATASET.get_count_train(), 
                         x_preprocess=lambda x:((x / 255.) - 0.5) * 2)

