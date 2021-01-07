# -*- coding: utf-8 -*-  
'''
训练rpn

@author: luoyi
Created on 2021年1月5日
'''
import utils.logger_factory as logf
import utils.conf as conf
import models.rpn as rpn
import models.layers.rpn.preprocess as preprocess
import data.dataset_rois as rois


log = logf.get_logger('train_rpn')


#    rpn 模型
log.info('init rpn_model...')
rpn_model = rpn.RPNModel(cnns_name=conf.RPN.get_cnns(), 
                         scaling=conf.CNNS.get_feature_map_scaling(), 
                         train_cnns=True,
                         train_rpn=True,
                         loss_lamda=conf.RPN.get_loss_lamda())
log.info('rpn_model finished.')
log.info('rpn_model cnns_name:%s scaling:%d loss_lamda:%f', conf.RPN.get_cnns(), conf.CNNS.get_feature_map_scaling(), conf.RPN.get_loss_lamda())



log.info('training RPNModel begin...')
#    准备数据集
db_train = rois.rpn_train_db(image_dir=conf.DATASET.get_in_train(), 
                             rois_out=conf.RPN.get_train_rois_out(), 
                             count_positives=conf.RPN.get_train_positives_every_image(),
                             count_negative=conf.RPN.get_train_negative_every_image(),
                             batch_size=conf.RPN.get_train_batch_size(),
                             ymaps_shape=rpn_model.rpn.get_output_shape(),
                             y_preprocess=lambda y:preprocess.preprocess_like_fmaps(y, shape=rpn_model.rpn.get_output_shape()))
log.info('db_train rois finished.')
log.info('db_train rois image_dir:%s', conf.DATASET.get_in_train())
log.info('db_train rois rois_out:%s', conf.RPN.get_train_rois_out())
log.info('db_train rois count_positives:%d count_negative:%d batch_size:%d', conf.RPN.get_train_positives_every_image(), conf.RPN.get_train_negative_every_image(), conf.RPN.get_train_batch_size())
log.info('db_train:{}'.format(db_train))


db_val = rois.rpn_train_db(image_dir=conf.DATASET.get_in_val(), 
                           rois_out=conf.RPN.get_val_rois_out(), 
                           count_positives=conf.RPN.get_train_positives_every_image(),
                           count_negative=conf.RPN.get_train_negative_every_image(),
                           batch_size=conf.RPN.get_train_batch_size(),
                           ymaps_shape=rpn_model.rpn.get_output_shape(),
                           y_preprocess=lambda y:preprocess.preprocess_like_fmaps(y, shape=rpn_model.rpn.get_output_shape()))
log.info('db_val rois finished.')
log.info('db_val rois image_dir:%s', conf.DATASET.get_in_val())
log.info('db_val rois rois_out:%s', conf.RPN.get_val_rois_out())
log.info('db_val rois count_positives:%d count_negative:%d batch_size:%d', conf.RPN.get_train_positives_every_image(), conf.RPN.get_train_negative_every_image(), conf.RPN.get_train_batch_size())
log.info('db_val:{}'.format(db_train))


log.info('rpn_model fitting...')
#    喂数据
rpn_model.train_tensor_db(db_train, db_val, 
                          batch_size=conf.RPN.get_train_batch_size(), 
                          epochs=conf.RPN.get_train_epochs(), 
                          auto_save_weights_after_traind=True, 
                          auto_save_weights_dir=conf.RPN.get_save_weights_dir(), 
                          auto_learning_rate_schedule=True, 
                          auto_tensorboard=True, 
                          auto_tensorboard_dir=conf.RPN.get_tensorboard_dir())


log.info('training RPNModel finished...')

