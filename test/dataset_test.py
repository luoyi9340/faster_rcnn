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

import utils.conf as conf
import data.dataset_rois as rois
from models.layers.rpn.preprocess import preprocess_like_array, takeout_sample_array


image_dir = conf.DATASET.get_in_train()
count = conf.DATASET.get_count_train()
rois_out = conf.ROIS.get_train_rois_out()
is_rois_mutiple_file = False
count_positives = conf.ROIS.get_positives_every_image()
count_negative = conf.ROIS.get_negative_every_image()
batch_size = conf.ROIS.get_batch_size()

#    验证数据集生成过程
db_iter = rois.read_rois_generator(count, rois_out, is_rois_mutiple_file, image_dir, count_positives, count_negative, batch_size, 
                                   x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                   y_preprocess=lambda y:preprocess_like_array(y))
db_train = rois.rpn_train_db(image_dir=image_dir, 
                             count=count, 
                             rois_out=rois_out, 
                             is_rois_mutiple_file=is_rois_mutiple_file, 
                             count_positives=count_positives, 
                             count_negative=count_negative, 
                             batch_size=batch_size, 
                             shuffle_buffer_rate=-1, 
                             epochs=1, 
                             ymaps_shape=(count_positives+count_negative, 10), 
                             x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                             y_preprocess=lambda y:preprocess_like_array(y))

x_cls = tf.random.uniform(shape=(batch_size, 23,60, 2, conf.ROIS.get_K()))
x_cls = tf.nn.softmax(x_cls, axis=3)
x_reg_dx = tf.ones(shape=(batch_size, 23,60, 1, conf.ROIS.get_K()))
x_reg_dy = tf.ones(shape=(batch_size, 23,60, 1, conf.ROIS.get_K())) * 2
x_reg_dw = tf.ones(shape=(batch_size, 23,60, 1, conf.ROIS.get_K())) * 3
x_reg_dh = tf.ones(shape=(batch_size, 23,60, 1, conf.ROIS.get_K())) * 4
x_reg = tf.concat([x_reg_dx, x_reg_dy, x_reg_dw, x_reg_dh], axis=3)
x = tf.concat([x_cls, x_reg], axis=3)


#    检查数量不对的到底什么情况
def checkout(anchors, P, N, y):
    print(y[3])
    pass


i_b = 0
for _, y in db_train:
    anchors = takeout_sample_array(y, x)
    #    取正负样本数
    anchors_p = anchors[anchors[:,:,0] > 0]
    anchors_n = anchors[anchors[:,:,0] < 0]
    P = tf.cast(tf.math.count_nonzero(anchors_p[:,0]), dtype=tf.float32) 
    N = tf.cast(tf.math.count_nonzero(anchors_n[:,0]), dtype=tf.float32) 
    P = int(P.numpy())
    N = int(N.numpy())
    if (P != count_positives * batch_size \
        or N != count_negative * batch_size):
        checkout(anchors, P, N, y)
        break
        pass
    i_b += 1
    pass

print(i_b)

