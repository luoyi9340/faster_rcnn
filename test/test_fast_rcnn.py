# -*- coding: utf-8 -*-  
'''
Fast rcnn测试程序

@author: luoyi
Created on 2021年2月5日
'''
import tensorflow as tf

import utils.conf as conf
import data.dataset_proposals as ds_proposals
from models.fast_rcnn import FastRcnnModel
from models.layers.fast_rcnn.preprocess import preprocess_like_array

#    加载训练模型时的配置
_, F_dataset, F_rois, F_rpn, F_cnns, F_context, F_proposales, F_fast_rcnn = conf.load_conf_yaml('/Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/temp/models/fast_rcnn/conf_FastRcnnModel.yml')


batch_size = conf.PROPOSALES.get_batch_size()
epochs = conf.PROPOSALES.get_epochs()
total_samples = ds_proposals.total_samples(proposal_out=conf.PROPOSALES.get_train_proposal_out(), 
                                           is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                           count=conf.DATASET.get_count_train(), 
                                           proposal_every_image=conf.PROPOSALES.get_proposal_every_image())
train_count = conf.DATASET.get_count_train()
test_count = conf.DATASET.get_count_test()
test_count = 10

#    加载训练数据集（其实没必要了）
_, train_y_crt_queue = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_train(), 
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

#    加载测试数据集
db_test, test_y_crt_queue = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_test(), 
                                                              count=test_count, 
                                                              proposals_out=conf.PROPOSALES.get_test_proposal_out(), 
                                                              is_proposal_mutiple_file=conf.DATASET.get_label_test_mutiple(), 
                                                              proposal_every_image=conf.PROPOSALES.get_proposal_every_image(), 
                                                              batch_size=batch_size, 
                                                              epochs=1, 
                                                              shuffle_buffer_rate=-1, 
                                                              ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9), 
                                                              x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                                              y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()))


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
                                train_ycrt_queue=train_y_crt_queue,
                                untrain_ycrt_queue=test_y_crt_queue)
fast_rcnn_model.load_model_weight('/Users/irenebritney/Desktop/workspace/eclipse-workspace2/faster_rcnn/temp/models/fast_rcnn/FastRcnnModel_10_54.99.h5')
fast_rcnn_model.cnns.trainable = False
fast_rcnn_model.fast_rcnn.trainable = False
fast_rcnn_model.show_info()

#    组装X, Y数据
X = []
Y = []
for x, y in db_test:
    X.append(x)
    Y.append(y)
    pass
X = tf.concat(X, axis=0)
Y = tf.concat(Y, axis=0)
print(X.shape) 
print(Y.shape)
#    计算预测结果
y_pred = fast_rcnn_model.test(db_test, batch_size)
#    检测分类准确率
acc = fast_rcnn_model.test_cls(Y, y_pred)
print('分类平均准确率:', acc.numpy(), ' 样本数:', test_count)
#    检测回归精度
mae_x, mae_y, mae_w, mae_h, m = fast_rcnn_model.test_reg(Y, y_pred)
print('mae:', m.numpy(), ' mae_x:', mae_x.numpy(), ' mae_y:', mae_y.numpy(), ' mae_w:', mae_w.numpy(), ' mae_h:', mae_h.numpy())

#    在图片上标出预测结果
