# -*- coding: utf-8 -*-  
'''
微调rpn
    固定cnns层，微调rpn层参数
    
@author: luoyi
Created on 2021年1月8日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import data.dataset_rois as ds_rois
import utils.logger_factory as logf
import utils.conf as conf
import models.layers.rpn.preprocess as preprocess
from models.fast_rcnn import FastRcnnModel
from models.rpn import RPNModel

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
fast_rcnn_model.fast_rcnn.trainable = False
#    从fast_rcnn中拿cnns模型
fast_rcnn_model.show_info()


log.info('load rpn model...')
#    用cnns加载rpn模型
rpn_model = RPNModel(cnns=fast_rcnn_model.cnns, 
                     learning_rate=conf.RPN.get_train_learning_rate(),
                     scaling=conf.CNNS.get_feature_map_scaling(), 
                     train_cnns=False,
                     train_rpn=True,
                     loss_lamda=conf.RPN.get_loss_lamda(),
                     is_build=True)
rpn_model.show_info()


log.info('training RPNModel begin...')
#    一些全局参数
#    单epoch数据总数
total_samples = ds_rois.total_samples(is_rois_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                      count=conf.DATASET.get_count_train(), 
                                      rois_out=conf.ROIS.get_train_rois_out())
batch_size = conf.ROIS.get_batch_size()
epochs = conf.ROIS.get_epochs()
#    准备数据集
db_train = ds_rois.rpn_train_db(count=conf.DATASET.get_count_train(),
                                image_dir=conf.DATASET.get_in_train(), 
                                rois_out=conf.ROIS.get_train_rois_out(), 
                                is_rois_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                                count_positives=conf.ROIS.get_positives_every_image(),
                                count_negative=conf.ROIS.get_negative_every_image(),
                                shuffle_buffer_rate=conf.ROIS.get_shuffle_buffer_rate(),
                                batch_size=batch_size,
                                epochs=epochs,
                                ymaps_shape=(conf.ROIS.get_positives_every_image() + conf.ROIS.get_negative_every_image(), 10),
                                x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                                y_preprocess=lambda y:preprocess.preprocess_like_array(y))
log.info('db_train rois finished.')
log.info('db_train rois image_dir:%s', conf.DATASET.get_in_train())
log.info('db_train rois rois_out:%s', conf.ROIS.get_train_rois_out())
log.info('db_train rois total_samples:%d count_positives:%d count_negative:%d batch_size:%d', total_samples, conf.ROIS.get_positives_every_image(), conf.ROIS.get_negative_every_image(), conf.ROIS.get_batch_size())
log.info('db_train:{}'.format(db_train))


db_val = ds_rois.rpn_train_db(count=conf.DATASET.get_count_val(),
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


log.info('rpn_model fitting...')
steps_per_epoch = total_samples / batch_size
if (total_samples % batch_size > 0): steps_per_epoch += 1
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

