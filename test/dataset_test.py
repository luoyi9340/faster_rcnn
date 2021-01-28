# -*- coding: utf-8 -*-  
'''
训练loss中有数据集总数不对的情况，怀疑数据还是有问题。。。
这里主要测试：
    1 数据集生成过程y_preprocess
    2 y_true取anchors过程

@author: luoyi
Created on 2021年1月22日
'''
import tensorflow as tf
import math

import utils.conf as conf
import data.dataset_rois as ds_rois
import data.dataset_proposals as ds_proposals
from models.layers.fast_rcnn.preprocess import preprocess_like_array


batch_size = 2
epochs = 2
count = 100
steps_per_epoch = math.ceil(count / batch_size)
print(steps_per_epoch)
# db_train = ds_proposals.fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_train(), 
#                                             count=count, 
#                                             proposals_out=conf.PROPOSALES.get_train_proposal_out(), 
#                                             is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
#                                             proposal_every_image=conf.PROPOSALES.get_proposal_every_image(), 
#                                             batch_size=batch_size, 
#                                             epochs=epochs, 
#                                             shuffle_buffer_rate=conf.PROPOSALES.get_shuffle_buffer_rate(), 
#                                             ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9), 
#                                             x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
#                                             y_preprocess=lambda y:preprocess_like_array(y, feature_map_scaling=conf.CNNS.get_feature_map_scaling()))
# db_iter = iter(db_train)
# step = 0
# for step in range(steps_per_epoch):
#     d = db_iter.next()
#     print(d[0].shape)
#     print(d[1].shape)
#     pass
# print(step)

def range_generator(count=100):
    for i in range(count):
        yield i, i
        pass
    pass
db = tf.data.Dataset.from_generator(generator=lambda :range_generator(count=150), 
                                    output_types=(tf.int32, tf.int32), 
                                    output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
db = db.batch(batch_size)
db = db.repeat(epochs)
db_iter = iter(db)
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        x, y = db_iter.next()
        print(epoch, step, x, y)
        pass
    pass



