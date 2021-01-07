# -*- coding: utf-8 -*-  
'''
通过原始图片和标签label.json生成训练rpn所需的数据集

@author: luoyi
Created on 2021年1月5日
'''
import main.main_path

import data.dataset_rois as rois
import utils.conf as conf
import utils.logger_factory as logf


log = logf.get_logger('data_rois_creator')

#    rois生成器
rois_creator = rois.RoisCreator()

#    生成训练rois
(num_anchors, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_train(), 
                                              rois_out=conf.RPN.get_train_rois_out(), 
                                              count=conf.DATASET.get_count_train())
log.info('create train_rois. num_anchors:%d num_positives:%d num_negative:%d', num_anchors, num_positives, num_negative)

#    生成验证rois
(num_anchors, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_val(), 
                                              rois_out=conf.RPN.get_val_rois_out(), 
                                              count=conf.DATASET.get_count_train())
log.info('create val_rois. num_anchors:%d num_positives:%d num_negative:%d', num_anchors, num_positives, num_negative)

#    生成测试rois
(num_anchors, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_test(), 
                                              rois_out=conf.RPN.get_test_rois_out(), 
                                              count=conf.DATASET.get_count_train())
log.info('create test_rois. num_anchors:%d num_positives:%d num_negative:%d', num_anchors, num_positives, num_negative)

