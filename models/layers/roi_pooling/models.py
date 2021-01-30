# -*- coding: utf-8 -*-  
'''
poi_pooling层
    包装一个layer出来是为了让模型直接连续

@author: luoyi
Created on 2021年1月12日
'''
import tensorflow as tf

import utils.conf as conf
from models.layers.roi_pooling.preprocess import roi_pooling


#    poi pooling层
class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, 
                 name=None, 
                 kernel_size=conf.FAST_RCNN.get_roipooling_kernel_size(), 
                 train_ycrt_queue=None, 
                 untrain_ycrt_queue=None,
                 input_shape=None,
                 output_shape=None,
                 **kwargs):
        '''roi pooling层
            @param kernel_size: 输出的特征图
            @param train_ycrt_queue: 训练时当前批次的y数据。list (batch_size)
                                        每个对象:tensor(num, 9)
            @param untrain_ycrt_queue: 非训练时当前批次的y数据
        '''
        #    该layer为dynamic layer。否则编译后两个queue的值不会更新
        super(ROIPooling, self).__init__(name=name, input_shape=input_shape, dynamic=True, **kwargs)
        
        self.__kernel_size = kernel_size
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        pass
    
    #    前向
    def call(self, x, training=None, **kwargs):
        #    如果是训练阶段
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
#         tf.print('training:', training)
#         tf.print('in roipooling y:', y)
        return roi_pooling(x, y, roipooling_ksize=self.__kernel_size)
    
    pass