# -*- coding: utf-8 -*-  
'''
训练loss中有数据集总数不对的情况，怀疑数据还是有问题。。。
这里主要测试：
    1 数据集生成过程y_preprocess
    2 y_true取anchors过程

@author: luoyi
Created on 2021年1月22日
'''
import numpy as np

import data.dataset as ds
import data.dataset_rois as ds_rois
import utils.conf as conf
import models.layers.rpn.preprocess as preprocess


# y = np.ones(shape=[2,5], dtype=np.float32)
# y = np.concatenate([y, -np.ones(shape=[6 - y.shape[0], 5], dtype=np.float32)])
# y = y[y[:,0] > 0]
# print(len(y))



db_train, rois_queue, y_queue = ds_rois.rpn_train_db_with_roisqueue_yqueue(count=conf.DATASET.get_count_train(),
                                                    image_dir=conf.DATASET.get_in_train(), 
                                                    rois_out=conf.ROIS.get_train_rois_out(), 
                                                    is_rois_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                                                    count_positives=conf.ROIS.get_positives_every_image(),
                                                    count_negative=conf.ROIS.get_negative_every_image(),
                                                    shuffle_buffer_rate=conf.ROIS.get_shuffle_buffer_rate(),
                                                    batch_size=2,
                                                    epochs=1,
                    #                                  ymaps_shape=rpn_model.rpn.get_output_shape(),
                                                    ymaps_shape=(conf.ROIS.get_positives_every_image() + conf.ROIS.get_negative_every_image(), 10),
                                                    x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                                                    y_preprocess=lambda y:preprocess.preprocess_like_array(y))

for x,y in db_train:
    print()
    print(x.shape)
    print(y.shape)
    print(rois_queue.crt_data().shape)
    print(y_queue.crt_data().shape)
    pass

