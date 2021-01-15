# -*- coding: utf-8 -*-  
'''
数据集测试

@author: luoyi
Created on 2020年12月29日
'''
import tensorflow as tf

import data.dataset_rois as rois
import utils.conf as conf
import models.rpn as rpn
import models.layers.rpn.preprocess as preprocess


# #    rpn 模型
# rpn_model = rpn.RPNModel(cnns_name=conf.RPN.get_cnns(), 
#                          scaling=conf.CNNS.get_feature_map_scaling(), 
#                          train_cnns=True,
#                          train_rpn=True,
#                          loss_lamda=conf.RPN.get_loss_lamda())
# 
# 
# 
# #    准备数据集
# db_train = rois.rpn_train_db(image_dir=conf.DATASET.get_in_train(), 
#                              rois_out=conf.RPN.get_train_rois_out(), 
#                              count_positives=conf.RPN.get_train_positives_every_image(),
#                              count_negative=conf.RPN.get_train_negative_every_image(),
#                              batch_size=conf.RPN.get_train_batch_size(),
#                              ymaps_shape=rpn_model.rpn.get_output_shape(),
#                              y_preprocess=lambda y:preprocess.preprocess_like_fmaps(y, shape=rpn_model.rpn.get_output_shape()))
# 
# for x, y in db_train:
# #     print(x)
# #     print(y)
#     pass
iter = rois.read_rois_generator(count=5, 
                         rois_out=conf.ROIS.get_train_rois_out(), 
                         is_rois_mutiple_file=True, 
                         image_dir=conf.DATASET.get_in_train(), 
                         count_positives=4, 
                         count_negative=4)
for x,y in iter:
    print()    


