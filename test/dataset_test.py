# -*- coding: utf-8 -*-  
'''
数据集测试

@author: luoyi
Created on 2020年12月29日
'''
import tensorflow as tf
import numpy as np

import data.dataset_rois as rois
import utils.conf as conf
import models.rpn as rpn
import models.layers.rpn.preprocess as preprocess

#    print时不用科学计数法表示
np.set_printoptions(suppress=True)     


image_dir = conf.DATASET.get_in_train()
count = conf.DATASET.get_count_train()
count = 1
rois_out = conf.ROIS.get_train_rois_out()
is_rois_mutiple_file = conf.DATASET.get_label_train_mutiple()
count_positives = conf.ROIS.get_positives_every_image()
count_negative = conf.ROIS.get_negative_every_image()
batch_size = conf.ROIS.get_batch_size()
epochs = conf.ROIS.get_epochs()
db = rois.rpn_train_db(image_dir=image_dir, 
                       count=count, 
                       rois_out=rois_out, 
                       is_rois_mutiple_file=is_rois_mutiple_file, 
                       count_positives=count_positives, 
                       count_negative=count_negative, 
                       batch_size=batch_size, 
                       shuffle_buffer_rate=-1, 
                       epochs=epochs, 
                       ymap_shape=10, 
                       x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                       y_preprocess=lambda y:preprocess.preprocess_like_array(y))

c = 0
for x, y in db:
    print(y)
    c += 1
    pass

print(c)

