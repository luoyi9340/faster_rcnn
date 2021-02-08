# -*- coding: utf-8 -*-  
'''
微调fast_rcnn
    固定cnns层，微调fast_rcnn层参数

@author: luoyi
Created on 2021年2月8日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import math

import data.dataset_proposals as ds_proposals
import utils.logger_factory as logf
import utils.conf as conf
from models.fast_rcnn import FastRcnnModel
from models.layers.fast_rcnn.preprocess import preprocess_like_array

log = logf.get_logger('rpn_fine_tuning')


log.info('load fast_rcnn model...')
#    加载训练模型时的配置
conf_path = conf.FAST_RCNN.get_save_weights_dir() + '/' + conf.FAST_RCNN.get_model_conf()
_, F_dataset, F_rois, F_rpn, F_cnns, F_context, F_proposales, F_fast_rcnn = conf.load_conf_yaml(conf_path)
model_path = conf.FAST_RCNN.get_save_weights_dir() + '/' + conf.FAST_RCNN.get_model_path()
fast_rcnn_model = FastRcnnModel(learning_rate=F_fast_rcnn.get_train_learning_rate(),
                                pooling=F_fast_rcnn.get_pooling(),
                                cnns_name=F_fast_rcnn.get_cnns(),
                                scaling=F_cnns.get_feature_map_scaling(), 
                                cnns_base_channel_num=F_cnns.get_base_channel_num(),
                                fc_weights=F_fast_rcnn.get_fc_weights(),
                                fc_dropout=F_fast_rcnn.get_fc_dropout(),
                                roipooling_ksize=F_fast_rcnn.get_roipooling_kernel_size(),
                                loss_lamda=F_fast_rcnn.get_loss_lamda(),
                                train_cnns=True,
                                train_fast_rcnn=True,
                                is_build=True,
                                input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
#                                 train_ycrt_queue=None,
#                                 untrain_ycrt_queue=None,
                                )
fast_rcnn_model.load_model_weight(model_path)
fast_rcnn_model.cnns.trainable = False
#    从fast_rcnn中拿cnns模型
fast_rcnn_model.show_info()


log.info('load train db...')
#    批量大小、epoch轮数、总样本数
batch_size = conf.PROPOSALES.get_batch_size()
epochs = conf.PROPOSALES.get_epochs()
total_samples = ds_proposals.total_samples(proposal_out=conf.PROPOSALES.get_train_proposal_out(), 
                                           is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                           count=conf.DATASET.get_count_train(), 
                                           proposal_every_image=conf.PROPOSALES.get_proposal_every_image())
train_count = conf.DATASET.get_count_train()
# train_count = 100
val_count = conf.DATASET.get_count_val()
# val_count = 50
log.info('total_samples:%d epochs:%d batch_size:%d', total_samples, epochs, batch_size)
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
                                                                y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()))
log.info('load train finished.')


log.info('load val db...')
db_val, val_y_crt_queue = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_val(), 
                                                              count=val_count, 
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


log.info('fast_rcnn training begin...')
steps_per_epoch = math.ceil(total_samples / batch_size)
fast_rcnn_model.train_tensor_db(db_train=db_train, 
                                db_val=db_val, 
                                steps_per_epoch=steps_per_epoch, 
                                batch_size=batch_size, 
                                epochs=epochs, 
                                auto_save_weights_after_traind=True, 
                                auto_save_weights_dir=conf.FAST_RCNN.get_save_weights_dir(), 
                                auto_learning_rate_schedule=True, 
                                auto_tensorboard=True, 
                                auto_tensorboard_dir=conf.FAST_RCNN.get_tensorboard_dir())
log.info('fast_rcnn training finished.')



