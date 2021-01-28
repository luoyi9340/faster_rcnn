# -*- coding: utf-8 -*-  
'''
训练fast_rcnn

@author: luoyi
Created on 2021年1月28日
'''
import math
import numpy as np
np.set_printoptions(suppress=True, threshold=16)

import utils.conf as conf
import utils.logger_factory as logf
import data.dataset_proposals as ds_proposals
from models.layers.fast_rcnn.preprocess import preprocess_like_array
from models.fast_rcnn import FastRcnnModel


log = logf.get_logger('fast_rcnn_train')


#    批量大小、epoch轮数、总样本数
batch_size = conf.PROPOSALES.get_batch_size()
epochs = conf.PROPOSALES.get_epochs()
total_samples = ds_proposals.total_samples(proposal_out=conf.PROPOSALES.get_train_proposal_out(), 
                                           is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                           count=conf.DATASET.get_count_train(), 
                                           proposal_every_image=conf.PROPOSALES.get_proposal_every_image())
log.info('total_samples:%d epochs:%d batch_size:%d', total_samples, epochs, batch_size)


log.info('load train db...')
db_train = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_train(), 
                                            count=conf.DATASET.get_count_train(), 
                                            proposals_out=conf.PROPOSALES.get_train_proposal_out(), 
                                            is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                            proposal_every_image=conf.PROPOSALES.get_proposal_every_image(), 
                                            batch_size=batch_size, 
                                            epochs=epochs, 
                                            shuffle_buffer_rate=conf.PROPOSALES.get_shuffle_buffer_rate(), 
                                            ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9), 
                                            x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                            y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()))
log.info('load train finished.')


log.info('load val db...')
db_val = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_val(), 
                                          count=conf.DATASET.get_count_val(), 
                                          proposals_out=conf.PROPOSALES.get_val_proposal_out(), 
                                          is_proposal_mutiple_file=conf.DATASET.get_label_val_mutiple(), 
                                          proposal_every_image=conf.PROPOSALES.get_proposal_every_image(), 
                                          batch_size=batch_size, 
                                          epochs=1, 
                                          shuffle_buffer_rate=-1, 
                                          ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9), 
                                          x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                          y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()))
log.info('load val db finished.')


log.info('load fast_rcnn model...')
fast_rcnn_model = FastRcnnModel(cnns_name=conf.FAST_RCNN.get_cnns(),
                                scaling=conf.CNNS.get_feature_map_scaling(), 
                                cnns_base_channel_num=conf.CNNS.get_base_channel_num(),
                                fc_weights=conf.FAST_RCNN.get_fc_weights(),
                                fc_layers=conf.FAST_RCNN.get_fc_layers(),
                                fc_dropout=conf.FAST_RCNN.get_fc_dropout(),
                                roipooling_ksize=conf.FAST_RCNN.get_roipooling_kernel_size(),
                                loss_lamda=conf.FAST_RCNN.get_loss_lamda(),
                                train_cnns=True,
                                train_fast_rcnn=True,
                                is_build=True,
                                input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3))
fast_rcnn_model.show_info()
log.info('load fast_rcnn model finished.')


log.info('fast_rcnn training begin...')
steps_per_epoch = math.ceil(total_samples / batch_size)
fast_rcnn_model.custom_train(db_train=db_train, 
                             db_val=db_val, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             steps_per_epoch=steps_per_epoch)
log.info('fast_rcnn training finished.')

