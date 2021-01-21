# -*- coding: utf-8 -*-  
'''
训练rpn

@author: luoyi
Created on 2021年1月5日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import numpy as np
np.set_printoptions(suppress=True, threshold=16)

import utils.logger_factory as logf
import utils.conf as conf
import models.rpn as rpn
import models.layers.rpn.preprocess as preprocess
import data.dataset_rois as rois


log = logf.get_logger('rpn_train')

#    rpn 模型
log.info('init rpn_model...')
rpn_model = rpn.RPNModel(cnns_name=conf.RPN.get_cnns(), 
                         learning_rate=conf.RPN.get_train_learning_rate(),
                         scaling=conf.CNNS.get_feature_map_scaling(), 
                         train_cnns=True,
                         train_rpn=True,
                         loss_lamda=conf.RPN.get_loss_lamda(),
                         is_build=True)
log.info('rpn_model finished.')
log.info('rpn_model cnns_name:%s scaling:%d loss_lamda:%f', conf.RPN.get_cnns(), conf.CNNS.get_feature_map_scaling(), conf.RPN.get_loss_lamda())


#    一些全局参数
#    单epoch数据总数
total_anchors = rois.total_anchors(is_rois_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                       count=conf.DATASET.get_count_train(), 
                                       rois_out=conf.ROIS.get_train_rois_out())
batch_size = conf.ROIS.get_batch_size()
epochs = conf.ROIS.get_epochs()


log.info('training RPNModel begin...')
#    准备数据集
db_train = rois.rpn_train_db(count=conf.DATASET.get_count_train(),
                             image_dir=conf.DATASET.get_in_train(), 
                             rois_out=conf.ROIS.get_train_rois_out(), 
                             is_rois_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                             count_positives=conf.ROIS.get_positives_every_image(),
                             count_negative=conf.ROIS.get_negative_every_image(),
                             shuffle_buffer_rate=conf.ROIS.get_shuffle_buffer_rate(),
                             batch_size=batch_size,
                             epochs=epochs,
#                              ymaps_shape=rpn_model.rpn.get_output_shape(),
                             ymaps_shape=(conf.ROIS.get_positives_every_image() + conf.ROIS.get_negative_every_image(), 10),
                             x_preprocess=None,
                             y_preprocess=lambda y:preprocess.preprocess_like_array(y))
log.info('db_train rois finished.')
log.info('db_train rois image_dir:%s', conf.DATASET.get_in_train())
log.info('db_train rois rois_out:%s', conf.ROIS.get_train_rois_out())
log.info('db_train rois count_positives:%d count_negative:%d batch_size:%d', conf.ROIS.get_positives_every_image(), conf.ROIS.get_negative_every_image(), conf.ROIS.get_batch_size())
log.info('db_train:{}'.format(db_train))


db_val = rois.rpn_train_db(count=conf.DATASET.get_count_val(),
                           image_dir=conf.DATASET.get_in_val(), 
                           rois_out=conf.ROIS.get_val_rois_out(), 
                           is_rois_mutiple_file=conf.DATASET.get_label_val_mutiple(),
                           count_positives=conf.ROIS.get_positives_every_image(),
                           count_negative=conf.ROIS.get_negative_every_image(),
                           batch_size=batch_size,
                           shuffle_buffer_rate=-1,
                           epochs=None,
                           ymaps_shape=(conf.ROIS.get_positives_every_image() + conf.ROIS.get_negative_every_image(), 10),
                           x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                           y_preprocess=lambda y:preprocess.preprocess_like_array(y))
log.info('db_val rois finished.')
log.info('db_val rois image_dir:%s', conf.DATASET.get_in_val())
log.info('db_val rois rois_out:%s', conf.ROIS.get_val_rois_out())
log.info('db_val rois count_positives:%d count_negative:%d batch_size:%d', conf.ROIS.get_positives_every_image(), conf.ROIS.get_negative_every_image(), conf.ROIS.get_batch_size())
log.info('db_val:{}'.format(db_train))


rpn_model.show_info()


log.info('rpn_model fitting...')
steps_per_epoch = total_anchors / batch_size
if (total_anchors % batch_size > 0): steps_per_epoch += 1
#    喂数据
rpn_model.train_tensor_db(db_train, db_val, 
                          batch_size=batch_size, 
                          epochs=epochs, 
                          steps_per_epoch=steps_per_epoch,
                          auto_save_weights_after_traind=True, 
                          auto_save_weights_dir=conf.RPN.get_save_weights_dir(), 
                          auto_learning_rate_schedule=True, 
                          auto_tensorboard=True, 
                          auto_tensorboard_dir=conf.RPN.get_tensorboard_dir())

#rpn_model.save_model_weights(conf.RPN.get_save_weights_dir() + '/rpn_resnet34.h5')

log.info('training RPNModel finished...')


