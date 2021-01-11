# -*- coding: utf-8 -*-  
'''
通过原始图片和标签label.json生成训练rpn所需的数据集

@author: luoyi
Created on 2021年1月5日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)


import data.dataset_rois as rois
import utils.conf as conf
import utils.logger_factory as logf


log = logf.get_logger('data_rois_creator')

#    rois生成器
rois_creator = rois.RoisCreator()

#    生成训练rois
(num_labels, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_train(), 
                                                                  rois_out=conf.ROIS.get_train_rois_out(), 
                                                                  count=conf.DATASET.get_count_train(),
                                                                  label_mutiple=conf.DATASET.get_label_train_mutiple(),
                                                                  max_workers=conf.ROIS.get_train_max_workers())
log.info('create train_rois. num_labels:%d num_positives:%d num_negative:%d', num_labels, num_positives, num_negative)

#    生成验证rois
(num_labels, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_val(), 
                                                                  rois_out=conf.ROIS.get_val_rois_out(), 
                                                                  count=conf.DATASET.get_count_train(),
                                                                  label_mutiple=conf.DATASET.get_label_val_mutiple(),
                                                                  max_workers=conf.ROIS.get_val_max_workers())
log.info('create val_rois. num_labels:%d num_positives:%d num_negative:%d', num_labels, num_positives, num_negative)

#    生成测试rois
(num_labels, num_positives, num_negative) = rois_creator.create(label_file_path=conf.DATASET.get_label_test(), 
                                                                  rois_out=conf.ROIS.get_test_rois_out(), 
                                                                  count=conf.DATASET.get_count_train(),
                                                                  label_mutiple=conf.DATASET.get_label_test_mutiple(),
                                                                  max_workers=conf.ROIS.get_test_max_workers())
log.info('create test_rois. num_labels:%d num_positives:%d num_negative:%d', num_labels, num_positives, num_negative)

