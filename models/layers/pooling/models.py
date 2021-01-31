# -*- coding: utf-8 -*-  
'''
poi_pooling层
    包装一个layer出来是为了让模型直接连续

@author: luoyi
Created on 2021年1月12日
'''
import tensorflow as tf

import utils.conf as conf
from models.layers.pooling.preprocess import roi_align


#    poi align层
class ROIAlign(tf.keras.layers.Layer):
    def __init__(self, 
                 name='ROIAlign', 
                 kernel_size=conf.FAST_RCNN.get_roipooling_kernel_size(), 
                 train_ycrt_queue=None, 
                 untrain_ycrt_queue=None,
                 input_shape=None,
                 **kwargs):
        '''roi pooling层
            @param kernel_size: 输出的特征图
            @param train_ycrt_queue: 训练时当前批次的y数据。list (batch_size)
                                        每个对象:tensor(num, 9)
            @param untrain_ycrt_queue: 非训练时当前批次的y数据
        '''
        #    该layer为dynamic layer。否则编译后两个queue的值不会更新
        super(ROIAlign, self).__init__(name=name, input_shape=input_shape, dynamic=True, **kwargs)
        
        self.__kernel_size = kernel_size
        self._train_ycrt_queue = train_ycrt_queue
        self._untrain_ycrt_queue = untrain_ycrt_queue
        pass
    
    #    编辑输出尺寸
    def compute_output_shape(self, input_shape):
        y_shape = None
        if (self._train_ycrt_queue): y_shape = self._train_ycrt_queue.crt_data().shape
        elif (self._untrain_ycrt_queue): y_shape = self._untrain_ycrt_queue.crt_data().shape
        #    取batch_size大小，每张图片proposal数量，cnns输出通道数
        _, _, C = y_shape[0], y_shape[1], input_shape[-1]
        #    roi_pooling的返回shap=(B * num, roi_ks[0], roi_ks[1], C)
        return (None, self.__kernel_size[0], self.__kernel_size[1], C)
    
    #    前向
    def call(self, x, training=None, **kwargs):
        #    如果是训练阶段
        if (training):
            y = self._train_ycrt_queue.crt_data()
            pass
        else:
            y = self._untrain_ycrt_queue.crt_data()
            pass
        return roi_align(x, y, roipooling_ksize=self.__kernel_size)
    
    pass